# Readout-only ablation for config/train_alopex_50m.py.
# Same model/data/head update, but no hidden FS-2T probes or hidden updates.
out_dir = 'out-head-only-50m'
dataset = 'openwebtext'
eval_interval = 50
eval_iters = 10
log_interval = 1
batch_size = 4
block_size = 128
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.0
bias = False
max_iters = 1000
dtype = 'bfloat16'
alopex_k = 4
alopex_rank = 64
alopex_lr = 1e-2
alopex_head_lr = 1e-2
alopex_consequence_mode = 'token'
alopex_hidden_enabled = False
