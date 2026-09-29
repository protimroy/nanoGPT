# Activation-Space ALOPEX on a Small Language Model

This experiment asks a deliberately narrow question:

> Can the repaired low-probe activation-space ALOPEX branch learn a causal language model when hidden credit is estimated with only `K=4` antithetic directions?

## Algorithm identity

This is **not** the frozen MNIST v43 configuration. Frozen v43 used complete hidden-space Hadamard coverage (`K=1024` for the 1024-wide reference). The SLM port follows the later repaired **v43-FS-2T / Two-Timescale Activation-Space ALOPEX** branch:

- hidden sensor: derivative-free antithetic activation perturbations;
- `K=4` simultaneous multi-layer probe pairs;
- persistent bilateral rank-64 credit state;
- `rho=0.95`, `nu=0.99`, actuator clip `0.08`;
- basis refresh every 8 updates;
- full stateless transient credit path in parallel with persistent low-rank credit.

The large-vocabulary LM head uses the historical **v43-hybrid pattern**: exact local cross-entropy output credit (`target - softmax`) rather than trying to reconstruct a ~50k-dimensional logit signal from four probes. Consequently, the default experiment is backprop-free in the hidden stack but is not scalar-query-only end-to-end. Because nanoGPT ties `lm_head.weight` to the token embedding table, this local readout update also changes the tied input embeddings; it does not include the separate input-embedding gradient contribution that backprop would compute.

No `backward()`, VJP, or JVP is used by `train_alopex.py`.

## Cost accounting

All target Transformer linears are perturbed **simultaneously**. The hidden sensor therefore needs `2K = 8` perturbed forwards per optimizer update, independent of Transformer depth. A separate unperturbed forward caches local inputs/preactivations and supplies the head readout, so the current implementation performs **9 total forward passes per update** at `K=4`.

Logs distinguish:

- **unique data tokens**: tokens drawn from the training stream once per update;
- **objective token evaluations**: unique tokens multiplied by the number of forwards;
- wall-clock step time.

This prevents a nine-forward ALOPEX update from being presented as compute-equivalent to one AdamW step.

## Consequence localization

`alopex_consequence_mode` is an explicit ablation:

- `sequence`: closest to the repaired FS-2T reference; one sequence-level consequence is broadcast across token positions;
- `token`: uses each next-token loss difference at the corresponding activation position;
- `suffix`: assigns token `t` the mean observed loss consequence over positions `t..T`.

`token` is the initial SLM candidate because language modeling exposes dense tokenwise losses. It is a language-model-specific extension and must not be conflated with the frozen v43 result.

## Run order

Prepare data with the normal nanoGPT scripts, then run the gates in order:

```bash
python data/shakespeare_char/prepare.py
python train_alopex.py config/train_alopex_shakespeare_char.py

python data/openwebtext/prepare.py
python train_alopex.py config/train_alopex_7m.py

# Only after the smaller BPE model shows sustained learning:
python train_alopex.py config/train_alopex_50m.py
python train_alopex.py config/train_head_only_50m.py
python train.py config/train_adamw_50m.py
```

For the 7M and 50M runs, repeat at least `sequence`, `token`, and `suffix` consequence modes before deciding which estimator scales best.

## Target model

The 50M config uses:

- 8 Transformer blocks;
- model width 512;
- 8 attention heads;
- context 128 for the first mechanism test;
- GPT-2-sized tied vocabulary table;
- approximately 50.9M nanoGPT parameters.

The short context is intentional. The first experiment is a credit-assignment test, not a competitive pretraining recipe.

## Evaluation contract

Primary quality curves:

1. training cross-entropy versus **unique tokens seen**;
2. validation cross-entropy/perplexity versus unique tokens seen;
3. validation cross-entropy versus **objective token evaluations**;
4. validation cross-entropy versus wall-clock time.

Mechanism diagnostics:

- hidden credit RMS;
- hidden update RMS;
- perturbation sigma;
- head credit/update RMS;
- probe consequence asymmetry;
- `K`, rank, and forward-evaluation count.

A diagnostic-only backprop run may later measure cosine alignment between ALOPEX activation credit and true activation gradients. Those gradients must never be fed into the ALOPEX training path.

## Interpretation gates

The first meaningful result is not “beats AdamW.” The gates are:

1. **Causal-LM sanity:** sustained loss reduction on the tiny Transformer without reverse-mode training.
2. **BPE scaling:** sustained learning on the ~7M model with `K=4`.
3. **50M viability:** sustained learning on the ~50.9M model without increasing `K` with parameter count.
4. **Hidden-credit contribution:** FS-2T must improve over the one-forward analytic-head-only ablation; otherwise falling LM loss is not evidence that the hidden estimator works.
5. **Competitiveness:** only then compare final quality, sample efficiency, objective-evaluation efficiency, memory, and wall-clock against AdamW.

A failure at any gate is informative: it identifies where four-probe activation credit stops carrying enough information.
