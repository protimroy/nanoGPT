"""Single-GPU nanoGPT training with activation-space ALOPEX v43-FS-2T.

The hidden Transformer linears are trained with simultaneous antithetic
activation probes and the repaired two-timescale FS-2T actuator. No backward
pass is used. The default LM head uses exact local cross-entropy credit, matching
the historical v43 hybrid pattern and avoiding four-probe reconstruction in a
~50k-dimensional vocabulary space.
"""

import math
import os
import pickle
import time

import numpy as np
import torch

from alopex_slm import AlopexSLMConfig, AlopexV43FS2TLM
from model import GPT, GPTConfig

# I/O
out_dir = "out-alopex"
eval_interval = 100
eval_iters = 20
log_interval = 1
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # scratch or resume

# data
dataset = "openwebtext"
batch_size = 4
block_size = 128

# model
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# run
max_iters = 1000
device = "cuda"
dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32"
seed = 1337

# ALOPEX v43-FS-2T hidden sensor / actuator
alopex_k = 4
alopex_sigma_rel = 0.03
alopex_sigma_min = 1e-4
alopex_rho = 0.95
alopex_nu = 0.99
alopex_u_max = 0.08
alopex_rank = 64
alopex_basis_refresh_every = 8
alopex_basis_alpha = 0.25
alopex_lambda_persistent = 1.0
alopex_lambda_transient = 1.0
alopex_lr = 1e-2
alopex_head_lr = 1e-2
alopex_consequence_mode = "sequence"  # sequence, token, suffix
alopex_label_smoothing = 0.0

# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

if int(os.environ.get("RANK", -1)) != -1:
    raise RuntimeError("train_alopex.py is intentionally single-GPU for the first mechanism test")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
forward_dtype = None if dtype == "float32" else ptdtype

data_dir = os.path.join("data", dataset)


def get_batch(split):
    path = os.path.join(data_dir, "train.bin" if split == "train" else "val.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        return (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    return x.to(device), y.to(device)


meta_vocab_size = None
meta_path = os.path.join(data_dir, "meta.pkl")
if os.path.exists(meta_path):
    with open(meta_path, "rb") as f:
        meta_vocab_size = pickle.load(f)["vocab_size"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

iter_num = 0
best_val_loss = 1e9
checkpoint = None
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=meta_vocab_size if meta_vocab_size is not None else 50304,
    dropout=dropout,
)

if init_from == "scratch":
    print("Initializing ALOPEX SLM from scratch")
    model = GPT(GPTConfig(**model_args))
elif init_from == "resume":
    checkpoint = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    saved_args = checkpoint["model_args"]
    for key in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[key] = saved_args[key]
    model = GPT(GPTConfig(**model_args))
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key, value in list(state_dict.items()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    iter_num = int(checkpoint["iter_num"])
    best_val_loss = float(checkpoint["best_val_loss"])
else:
    raise ValueError(f"unsupported init_from={init_from!r}")

model.to(device)
model.train()

alopex_cfg = AlopexSLMConfig(
    K=alopex_k,
    sigma_rel=alopex_sigma_rel,
    sigma_min=alopex_sigma_min,
    rho=alopex_rho,
    nu=alopex_nu,
    u_max=alopex_u_max,
    rank=alopex_rank,
    basis_refresh_every=alopex_basis_refresh_every,
    basis_alpha=alopex_basis_alpha,
    lambda_persistent=alopex_lambda_persistent,
    lambda_transient=alopex_lambda_transient,
    lr=alopex_lr,
    head_lr=alopex_head_lr,
    consequence_mode=alopex_consequence_mode,
    head_mode="analytic",
    label_smoothing=alopex_label_smoothing,
)
alopex = AlopexV43FS2TLM(model, alopex_cfg, forward_dtype=forward_dtype)
if checkpoint is not None and "alopex" in checkpoint:
    alopex.load_state_dict(checkpoint["alopex"])
checkpoint = None

print(
    "ALOPEX config:",
    f"K={alopex_k}",
    f"rank={alopex_rank}",
    f"rho={alopex_rho}",
    f"nu={alopex_nu}",
    f"consequence={alopex_consequence_mode}",
)
print(f"sensor forwards/update: {2 * alopex_k}; total forwards/update including base cache: {1 + 2 * alopex_k}")
print(f"unique data tokens/update: {batch_size * block_size:,}")


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with torch.amp.autocast(
                device_type=device_type,
                dtype=ptdtype,
                enabled=(forward_dtype is not None),
            ):
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


while True:
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses["val"])
            if iter_num > 0:
                ckpt = {
                    "model": model.state_dict(),
                    "alopex": alopex.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
    if iter_num == 0 and eval_only:
        break

    x, y = get_batch("train")
    if device_type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    metrics = alopex.step(x, y)
    if device_type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if iter_num % log_interval == 0:
        toks = metrics["unique_data_tokens"]
        objective_toks = metrics["objective_token_evals"]
        print(
            f"iter {iter_num}: loss {metrics['loss']:.4f}, "
            f"time {dt * 1000:.1f}ms, data tok/s {toks / dt:,.0f}, "
            f"objective tok/s {objective_toks / dt:,.0f}, "
            f"credit rms {metrics['hidden_credit_rms']:.3e}, "
            f"update rms {metrics['hidden_update_rms']:.3e}, "
            f"sigma {metrics['sigma_mean']:.3e}"
        )

    iter_num += 1
    if iter_num > max_iters:
        break

alopex.close()
