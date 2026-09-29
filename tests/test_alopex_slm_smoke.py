"""CPU smoke checks for the nanoGPT ALOPEX SLM port.

Run directly from the repository root:
    python tests/test_alopex_slm_smoke.py
"""

import torch

from alopex_slm import AlopexSLMConfig, AlopexV43FS2TLM
from model import GPT, GPTConfig


def tiny_model():
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=32,
            n_layer=2,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
        )
    )


def run_hidden_credit_smoke():
    torch.manual_seed(7)
    model = tiny_model()
    opt = AlopexV43FS2TLM(
        model,
        AlopexSLMConfig(
            K=2,
            rank=8,
            lr=1e-3,
            head_lr=1e-3,
            consequence_mode="suffix",
        ),
    )
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    metrics = opt.step(x, y)
    after = model.transformer.h[0].attn.c_attn.weight.detach()

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["forward_evals"] == 5.0
    assert not torch.equal(before, after), "FS-2T did not change a hidden Transformer weight"
    assert all(p.grad is None for p in model.parameters()), "reverse-mode gradients were populated"
    opt.close()


def run_head_only_ablation_smoke():
    torch.manual_seed(11)
    model = tiny_model()
    opt = AlopexV43FS2TLM(
        model,
        AlopexSLMConfig(
            K=2,
            rank=8,
            lr=1e-3,
            head_lr=1e-3,
            consequence_mode="token",
            hidden_enabled=False,
        ),
    )
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    hidden_before = model.transformer.h[0].attn.c_attn.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    metrics = opt.step(x, y)

    assert metrics["forward_evals"] == 1.0
    assert torch.equal(hidden_before, model.transformer.h[0].attn.c_attn.weight.detach())
    assert not torch.equal(head_before, model.lm_head.weight.detach())
    assert all(p.grad is None for p in model.parameters())
    opt.close()


if __name__ == "__main__":
    run_hidden_credit_smoke()
    run_head_only_ablation_smoke()
    print("ALOPEX SLM smoke checks passed")
