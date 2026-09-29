"""Activation-space ALOPEX / FS-2T optimizer for nanoGPT-style language models.

This module is an experimental SLM port of the repaired ALOPEX v43-FS-2T
mechanism. It deliberately avoids reverse-mode autodiff: no ``backward()``, VJP,
or JVP is used. Hidden linear preactivations are perturbed simultaneously with
K antithetic directions, so the sensor uses 2K perturbed model evaluations per
update rather than 2K evaluations per layer.

The default language-model head uses the exact local cross-entropy output credit
(target - softmax) as in the historical v43 hybrid readout. Therefore the default
configuration is *not* scalar-query-only end-to-end; the hidden sensor remains
derivative-free while the output head has an analytic local credit channel.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AlopexSLMConfig:
    # Derivative-free hidden sensor.
    K: int = 4
    sigma_rel: float = 0.03
    sigma_min: float = 1e-4

    # Modern two-timescale state.
    rho: float = 0.95
    nu: float = 0.99
    u_max: float = 0.08
    eps: float = 1e-8

    # Persistent bilateral credit subspace.
    rank: int = 64
    basis_refresh_every: int = 8
    basis_alpha: float = 0.25

    # Persistent/transient mixing and gain.
    lambda_persistent: float = 1.0
    lambda_transient: float = 1.0
    lr: float = 1e-2
    head_lr: Optional[float] = None

    # LM consequence localization.
    #   sequence: faithful per-example scalar consequence, broadcast over tokens.
    #   token:    use each token loss difference for the activation at that token.
    #   suffix:   use mean future loss difference from t..T for token t.
    consequence_mode: str = "sequence"

    # v43-hybrid-style readout. The large-vocabulary SLM experiment intentionally
    # avoids K=4 probing in 50k-dimensional logit space.
    head_mode: str = "analytic"
    label_smoothing: float = 0.0

    # Module selection. The head is handled separately.
    include_linear_prefix: str = "transformer.h."


@torch.no_grad()
def _random_orthobasis(n: int, r: int, *, device: torch.device, seed: int) -> torch.Tensor:
    r = min(n, r)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, r, generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(a, mode="reduced")
    return q.to(device)


class TwoTimescaleState:
    def __init__(self, shape, cfg: AlopexSLMConfig, *, device: torch.device):
        self.C = torch.zeros(shape, dtype=torch.float32, device=device)
        self.S = torch.zeros_like(self.C)
        self.rho = cfg.rho
        self.nu = cfg.nu

    @torch.no_grad()
    def update(self, signal: torch.Tensor, cfg: AlopexSLMConfig) -> torch.Tensor:
        signal = signal.float()
        self.C.mul_(self.rho).add_(signal, alpha=1.0 - self.rho)
        self.S.mul_(self.nu).addcmul_(self.C, self.C, value=1.0 - self.nu)
        return (self.C / (self.S.sqrt() + cfg.eps)).clamp_(-cfg.u_max, cfg.u_max)


class PersistentCreditSubspace:
    def __init__(
        self,
        out_dim: int,
        in_dim: int,
        cfg: AlopexSLMConfig,
        *,
        device: torch.device,
        seed: int,
    ):
        self.U = _random_orthobasis(out_dim, cfg.rank, device=device, seed=seed)
        self.V = _random_orthobasis(in_dim, cfg.rank, device=device, seed=seed + 1)
        self.state = TwoTimescaleState(
            (self.U.shape[1], self.V.shape[1]), cfg, device=device
        )

    @torch.no_grad()
    def projected_credit(self, g_out: torch.Tensor, h_in: torch.Tensor) -> torch.Tensor:
        n = max(1, h_in.shape[0])
        gu = g_out.float() @ self.U
        hv = h_in.float() @ self.V
        return gu.T @ hv / n

    @torch.no_grad()
    def persistent_update(
        self, g_out: torch.Tensor, h_in: torch.Tensor, cfg: AlopexSLMConfig
    ) -> torch.Tensor:
        m = self.projected_credit(g_out, h_in)
        p = self.state.update(m, cfg)
        return self.U @ p @ self.V.T

    @torch.no_grad()
    def refresh_basis(self, g_out: torch.Tensor, h_in: torch.Tensor, alpha: float) -> None:
        # Same bilinear subspace iteration and heuristic S transport as the repaired
        # FS-2T reference notebook.
        n = max(1, h_in.shape[0])
        gv = g_out.float().T @ (h_in.float() @ self.V) / n
        gtu = h_in.float().T @ (g_out.float() @ self.U) / n

        u_cand, _ = torch.linalg.qr(gv, mode="reduced")
        v_cand, _ = torch.linalg.qr(gtu, mode="reduced")

        u_old, v_old = self.U, self.V
        u_new, _ = torch.linalg.qr((1.0 - alpha) * u_old + alpha * u_cand, mode="reduced")
        v_new, _ = torch.linalg.qr((1.0 - alpha) * v_old + alpha * v_cand, mode="reduced")

        ru = u_old.T @ u_new
        rv = v_old.T @ v_new
        self.state.C = ru.T @ self.state.C @ rv
        self.state.S = (ru.T.abs() @ self.state.S @ rv.abs()).clamp_min_(0)
        self.U, self.V = u_new, v_new

    def state_dict(self) -> dict:
        return {
            "U": self.U,
            "V": self.V,
            "C": self.state.C,
            "S": self.state.S,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        self.U.copy_(state["U"])
        self.V.copy_(state["V"])
        self.state.C.copy_(state["C"])
        self.state.S.copy_(state["S"])


@torch.no_grad()
def normalize_transient_credit(g: torch.Tensor, cfg: AlopexSLMConfig) -> torch.Tensor:
    g32 = g.float()
    rms = g32.square().mean().sqrt()
    return (g32 / (rms + cfg.eps)).clamp_(-cfg.u_max, cfg.u_max)


@torch.no_grad()
def _rademacher_like(x: torch.Tensor) -> torch.Tensor:
    # Keep probe storage compact; accumulation is FP32.
    dtype = torch.bfloat16 if x.device.type == "cuda" else torch.float32
    return (
        torch.randint(0, 2, x.shape, device=x.device, dtype=torch.int8)
        .to(dtype)
        .mul_(2.0)
        .sub_(1.0)
    )


class AlopexV43FS2TLM:
    """Simultaneous multi-layer activation-space FS-2T for nanoGPT.

    Only Transformer ``nn.Linear`` modules under ``include_linear_prefix`` are
    perturbed and updated by FS-2T. The LM head is handled by a separate local
    readout channel. LayerNorm and positional-embedding parameters are frozen in
    this first SLM mechanism test.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: AlopexSLMConfig,
        *,
        forward_dtype: Optional[torch.dtype] = None,
    ):
        if cfg.K < 1:
            raise ValueError("K must be >= 1")
        if cfg.consequence_mode not in {"sequence", "token", "suffix"}:
            raise ValueError(f"unknown consequence_mode={cfg.consequence_mode!r}")
        if cfg.head_mode != "analytic":
            raise ValueError("initial SLM port supports head_mode='analytic' only")
        if hasattr(model, "config") and float(getattr(model.config, "dropout", 0.0)) != 0.0:
            raise ValueError("antithetic pairing requires dropout=0.0 in the initial SLM port")

        self.model = model
        self.cfg = cfg
        self.forward_dtype = forward_dtype
        self.device = next(model.parameters()).device
        self.step_idx = 0
        self.mode: Optional[str] = None
        self.cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.probe_delta: Dict[str, torch.Tensor] = {}
        self.modules: Dict[str, nn.Linear] = {}
        self.subspaces: Dict[str, PersistentCreditSubspace] = {}
        self._handles = []
        self._head_input: Optional[torch.Tensor] = None

        for p in self.model.parameters():
            p.requires_grad_(False)

        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if name == "lm_head":
                continue
            if not name.startswith(cfg.include_linear_prefix):
                continue
            self.modules[name] = module
            self.subspaces[name] = PersistentCreditSubspace(
                out_dim=module.out_features,
                in_dim=module.in_features,
                cfg=cfg,
                device=module.weight.device,
                seed=1000 + 17 * len(self.subspaces),
            )
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

        if not self.modules:
            raise ValueError("no Transformer Linear modules matched include_linear_prefix")

        if not hasattr(model, "lm_head"):
            raise ValueError("model must expose lm_head")
        self._handles.append(model.lm_head.register_forward_pre_hook(self._head_pre_hook))


    def _forward_context(self):
        if self.forward_dtype is None or self.device.type not in {"cuda", "cpu"}:
            return nullcontext()
        return torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.forward_dtype,
        )

    def _head_pre_hook(self, module, inputs):
        self._head_input = inputs[0].detach()

    def _make_hook(self, name: str):
        def hook(module, inputs, output):
            if self.mode == "base":
                self.cache[name] = (inputs[0].detach(), output.detach())
                return output
            if self.mode == "plus":
                return output + self.probe_delta[name].to(output.dtype)
            if self.mode == "minus":
                return output - self.probe_delta[name].to(output.dtype)
            return output

        return hook

    @staticmethod
    def _per_token_loss(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
        b, t, v = logits.shape
        return F.cross_entropy(
            logits.float().reshape(b * t, v),
            targets.reshape(b * t),
            reduction="none",
            ignore_index=-1,
            label_smoothing=label_smoothing,
        ).view(b, t)

    def _localize_consequence(self, diff: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets.ne(-1)
        if self.cfg.consequence_mode == "token":
            return diff * valid
        if self.cfg.consequence_mode == "sequence":
            denom = valid.sum(1, keepdim=True).clamp_min(1)
            per_sequence = (diff * valid).sum(1, keepdim=True) / denom
            return per_sequence.expand_as(diff)

        # suffix: token t receives the mean observed consequence over t..T.
        rev_diff = torch.flip(diff * valid, dims=[1])
        rev_valid = torch.flip(valid, dims=[1]).to(diff.dtype)
        suffix_sum = torch.flip(torch.cumsum(rev_diff, dim=1), dims=[1])
        suffix_n = torch.flip(torch.cumsum(rev_valid, dim=1), dims=[1]).clamp_min(1.0)
        return suffix_sum / suffix_n

    @torch.no_grad()
    def _analytic_head_update(
        self, logits: torch.Tensor, targets: torch.Tensor, head_input: torch.Tensor
    ) -> dict[str, float]:

        probs = F.softmax(logits.float(), dim=-1)
        favorable = -probs
        valid = targets.ne(-1)
        safe_targets = targets.masked_fill(~valid, 0)
        favorable.scatter_add_(
            -1,
            safe_targets.unsqueeze(-1),
            valid.unsqueeze(-1).to(favorable.dtype),
        )
        favorable.mul_(valid.unsqueeze(-1))

        if self.cfg.label_smoothing:
            # Exact local favorable credit for CE with uniform label smoothing.
            eps = self.cfg.label_smoothing
            favorable.mul_(1.0 - eps)
            favorable.add_(valid.unsqueeze(-1).to(favorable.dtype), alpha=eps / logits.size(-1))
            favorable.add_(probs * valid.unsqueeze(-1), alpha=-eps)

        h = head_input.float()
        n = max(1, int(valid.sum()))
        g2 = favorable.reshape(-1, favorable.shape[-1])
        h2 = h.reshape(-1, h.shape[-1])
        d_w = g2.T @ h2 / n
        lr = self.cfg.lr if self.cfg.head_lr is None else self.cfg.head_lr
        self.model.lm_head.weight.add_(lr * d_w.to(self.model.lm_head.weight.dtype))
        return {
            "head_update_rms": float((lr * d_w).square().mean().sqrt()),
            "head_credit_rms": float(favorable.square().mean().sqrt()),
        }

    @torch.no_grad()
    def step(self, idx: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        self.step_idx += 1
        self.cache.clear()

        # Base pass: cache local presynaptic inputs and preactivations, and obtain
        # the LM-head input for its exact local readout update.
        self.mode = "base"
        with self._forward_context():
            base_logits, _ = self.model(idx, return_all_logits=True)
        self.mode = None
        if self._head_input is None:
            raise RuntimeError("LM head input was not captured")
        base_head_input = self._head_input.detach()
        base_token_loss = self._per_token_loss(base_logits, targets, self.cfg.label_smoothing)
        valid_for_base = targets.ne(-1)
        base_loss = (base_token_loss * valid_for_base).sum() / valid_for_base.sum().clamp_min(1)
        if set(self.cache) != set(self.modules):
            missing = sorted(set(self.modules) - set(self.cache))
            raise RuntimeError(f"some target Linear modules did not execute: {missing[:4]}")

        sigmas = {
            name: max(
                self.cfg.sigma_min,
                self.cfg.sigma_rel * float(preact.float().square().mean().sqrt()),
            )
            for name, (_, preact) in self.cache.items()
        }
        credit = {
            name: torch.zeros_like(preact, dtype=torch.float32)
            for name, (_, preact) in self.cache.items()
        }

        plus_loss_sum = 0.0
        minus_loss_sum = 0.0
        for _ in range(self.cfg.K):
            xis = {name: _rademacher_like(preact) for name, (_, preact) in self.cache.items()}
            self.probe_delta = {name: sigmas[name] * xi for name, xi in xis.items()}

            self.mode = "plus"
            with self._forward_context():
                logits_p, _ = self.model(idx, return_all_logits=True)
            loss_p = self._per_token_loss(logits_p, targets, self.cfg.label_smoothing)

            self.mode = "minus"
            with self._forward_context():
                logits_m, _ = self.model(idx, return_all_logits=True)
            loss_m = self._per_token_loss(logits_m, targets, self.cfg.label_smoothing)
            self.mode = None

            diff = self._localize_consequence(loss_m - loss_p, targets)
            plus_loss_sum += float(loss_p.mean())
            minus_loss_sum += float(loss_m.mean())

            for name, xi in xis.items():
                coeff = diff / (2.0 * sigmas[name])
                credit[name].add_(coeff.unsqueeze(-1) * xi.float())

            del logits_p, logits_m, loss_p, loss_m, xis

        update_rms = []
        credit_rms = []
        for name, module in self.modules.items():
            h_in, _ = self.cache[name]
            g = (credit[name] / self.cfg.K).reshape(-1, module.out_features)
            h = h_in.float().reshape(-1, module.in_features)
            subspace = self.subspaces[name]

            if self.cfg.basis_refresh_every > 0 and self.step_idx % self.cfg.basis_refresh_every == 0:
                subspace.refresh_basis(g, h, alpha=self.cfg.basis_alpha)

            d_w_persistent = subspace.persistent_update(g, h, self.cfg)
            g_fresh = normalize_transient_credit(g, self.cfg)
            d_w_transient = g_fresh.T @ h / max(1, h.shape[0])
            d_w = (
                self.cfg.lambda_persistent * d_w_persistent
                + self.cfg.lambda_transient * d_w_transient
            )
            applied = self.cfg.lr * d_w
            module.weight.add_(applied.to(module.weight.dtype))
            if module.bias is not None:
                db = g_fresh.mean(0)
                module.bias.add_(
                    self.cfg.lr
                    * self.cfg.lambda_transient
                    * db.to(module.bias.dtype)
                )

            update_rms.append(applied.square().mean().sqrt())
            credit_rms.append(g.square().mean().sqrt())

        head_diag = self._analytic_head_update(base_logits, targets, base_head_input)
        self.probe_delta.clear()
        self.mode = None

        valid = targets.ne(-1)
        unique_tokens = int(valid.sum())
        forward_evals = 1 + 2 * self.cfg.K
        return {
            "loss": float(base_loss),
            "probe_plus_loss": plus_loss_sum / self.cfg.K,
            "probe_minus_loss": minus_loss_sum / self.cfg.K,
            "sigma_mean": sum(sigmas.values()) / len(sigmas),
            "hidden_credit_rms": float(torch.stack(credit_rms).mean()),
            "hidden_update_rms": float(torch.stack(update_rms).mean()),
            "head_update_rms": head_diag["head_update_rms"],
            "head_credit_rms": head_diag["head_credit_rms"],
            "K": float(self.cfg.K),
            "forward_evals": float(forward_evals),
            "unique_data_tokens": float(unique_tokens),
            "objective_token_evals": float(unique_tokens * forward_evals),
        }

    def state_dict(self) -> dict:
        return {
            "step_idx": self.step_idx,
            "config": asdict(self.cfg),
            "subspaces": {name: sub.state_dict() for name, sub in self.subspaces.items()},
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        self.step_idx = int(state["step_idx"])
        for name, sub_state in state["subspaces"].items():
            if name not in self.subspaces:
                raise KeyError(f"checkpoint has unknown subspace {name}")
            self.subspaces[name].load_state_dict(sub_state)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
