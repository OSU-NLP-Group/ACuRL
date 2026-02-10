1. 
Setup the vllm via:

```bash

vllm serve /fs/ess/PAA0201/tianci/Models/Qwen3-VL-8B-Instruct \
  --data-parallel-size 4 \
  --trust-remote-code \
  --limit-mm-per-prompt.video 0 \
  --max-model-len 8k \
  --max-num-batched-tokens 8k \
  --async-scheduling


vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --data-parallel-size 4 \
  --trust-remote-code \
  --limit-mm-per-prompt.video 0 \
  --max-model-len 8k \
  --max-num-batched-tokens 8k


vllm serve /fs/ess/PAA0201/tianci/Models/Qwen3-VL-8B-Instruct \
  --served-model-name qwen3-vl-8b \
  --data-parallel-size 4 \
  --trust-remote-code \
  --limit-mm-per-prompt.video 0 \
  --max-model-len 8k \
  --max-num-batched-tokens 8k

```

2.
choose 2.1 or 2.2 

2.1
change the `scripts/osworld/local/docker/iterative_training/init_software_model.sh` accordingly.

Set `use_vllm_for_key_screenshot` to True, if we use vllm for the second stage.


set `vllm_base_url` to the url where you setup the vllm.


2.2
It's okay to not change the `init_software_model.sh` but instead change the config directly.

You can add the following snippet into the config file:

```
"use_vllm_for_key_screenshot": true,
"vllm_base_url": "http://c0821:8000/v1",
"cuajudge_key_model": "Qwen/Qwen3-VL-8B-Instruct",
```

A example looks like this:

```
  "training": {
    "algorithm": "grpo",
    "learning_rate": "1e-6",
    "ppo_mini_batch_size": 8192,
    "ppo_micro_batch_size_per_gpu": 4,
    "use_kl_loss": false,
    "kl_loss_coef": 0.0,
    "clip_ratio_low": 0.2,
    "clip_ratio_high": 0.28,
    "clip_ratio_c": 10.0,
    "loss_agg_mode": "token-mean",
    "rollout_n": 8,
    "fsdp_param_offload": false,
    "fsdp_optimizer_offload": false,
    "ref_log_prob_micro_batch_size_per_gpu": 4,
    "rollout_log_prob_micro_batch_size_per_gpu": 4,
    "critic_warmup": 0,
    "save_freq": 5,
    "test_freq": 100,
    "total_training_steps": 75,
    "use_vllm_for_key_screenshot": true,
    "vllm_base_url": "http://c0821:8000/v1",
    "cuajudge_key_model": "Qwen/Qwen3-VL-8B-Instruct",
    "time_stamp": "20251220_222957" (**OPTIONAL**)
  }
