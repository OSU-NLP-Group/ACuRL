#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/iterative_curriculum/create_config.sh <software> <training_model_name> <exploration_model> <training_method> <out_config_json>

This writes a simple JSON config file (no auto-detection heuristics).
Edit the generated file to customize paths / defaults.
EOF
}

[[ $# -eq 5 ]] || { usage; exit 2; }

SOFTWARE="$1"
TRAINING_MODEL_NAME="$2"
EXPLORATION_MODEL="$3"
TRAINING_METHOD="$4"
OUT_CFG="$5"

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$(dirname "$OUT_CFG")"

if [[ -f "$OUT_CFG" ]]; then
  echo "[ACuRL] ERROR: refusing to overwrite existing config: $OUT_CFG" >&2
  exit 1
fi

cat >"$OUT_CFG" <<EOF
{
  "start_iter": 0,
  "max_iter": 3,

  "env_exploration_traj_dir": "./trajectory/UI-TARS-1.5-7B/environment_exploration/test/libreoffice_impress",
  "context_review_root_dir": "./trajectory/UI-TARS-1.5-7B/context_review/test/libreoffice_impress",

  "desktop_config_file": "data/config_examples/environment_config.json",

  "seed": 42,
  "group_size": 8,

  "mode": "visual",
  "train_data_size": 500,
  "val_data_size": 500,
  "task_copy_times": 1,

  "train_batch_size": 16,
  "val_batch_size": 128,

  "total_training_steps": 75,
  "save_freq": 5,
  "test_freq": -1,
  "n_gpus_per_node": 4,
  "nnodes": 2,

  "max_steps_train": 15,
  "max_steps_eval": 25,

  "max_steps_train_iter_1": 15,
  "max_steps_train_iter_2": 20,
  "max_steps_train_iter_3": 25,

  "max_prompt_len_train_iter_1": 13000,
  "max_prompt_len_train_iter_2": 15000,
  "max_prompt_len_train_iter_3": 17000,

  "max_steps_eval_iter_0": 15,
  "max_steps_eval_iter_1": 20,
  "max_steps_eval_iter_2": 25,

  "max_prompt_len_eval_iter_0": 13000,
  "max_prompt_len_eval_iter_1": 15000,
  "max_prompt_len_eval_iter_2": 17000,

  "learning_rate": 1e-6,
  "ppo_mini_batch_size": 8192,
  "ppo_micro_batch_size_per_gpu": 4,
  "use_kl_loss": false,
  "kl_loss_coef": 0.0,
  "kl_loss_type": "low_var_kl",
  "clip_ratio_low": 0.2,
  "clip_ratio_high": 0.28,
  "clip_ratio_c": 10.0,
  "loss_agg_mode": "token-mean",
  "rollout_log_prob_micro_batch_size_per_gpu": 4,
  "limit_images": 2,
  "ref_log_prob_micro_batch_size_per_gpu": 4,
  "use_kl_in_reward": false,
  "kl_ctrl_kl_coef": 0.0,
  "random_task_sampling": false,

  "cuajudge_use_vllm_for_key_screenshot": true,
  "cuajudge_vllm_base_url": "http://IP:8000/v1",
  "cuajudge_key_model": "YOUR_MODEL_NAME",

  "enable_cuajudge": true,
  "enable_cuajudge_train": true,
  "enable_cuajudge_eval": true,
  "cuajudge_outcome_model": "gpt-5-mini",
  "cuajudge_max_image": 50,
  "cuajudge_score_threshold": 3
}
EOF

echo "[ACuRL] wrote config: $OUT_CFG"

