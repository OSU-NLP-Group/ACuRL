#!/usr/bin/env bash
SOFTWARE="${1:-libreoffice_impress}"
TRAINING_MODEL_NAME="${2:-Qwen3-VL-8B-Instruct}"
EXPLORATION_MODEL="${3:-gpt-5}"
TRAINING_METHOD="${4:-ACuRL_training_256}"
PROJECT_NAME="ACuRL"

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export ROCR_VISIBLE_DEVICES=""

# Derived run name.
RUN_NAME="ACuRL_${SOFTWARE}_${TRAINING_MODEL_NAME}_${EXPLORATION_MODEL}_${TRAINING_METHOD}"

LOG_DIR="./logs/${TRAINING_MODEL_NAME}/ACuRL/"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

CFG_DIR="./checkpoints/${PROJECT_NAME}/${TRAINING_MODEL_NAME}/${RUN_NAME}"
mkdir -p "$CFG_DIR"
CFG_FILE="${CFG_DIR}/ACuRL_config.json"

# Auto-create config if missing (for clean release UX).
if [[ ! -f "$CFG_FILE" ]]; then
  echo "[ACuRL] config not found, creating: $CFG_FILE"
  bash "scripts/ACuRL/create_config.sh" "$SOFTWARE" "$TRAINING_MODEL_NAME" "$EXPLORATION_MODEL" "$TRAINING_METHOD" "$CFG_FILE"
  echo "[ACuRL] NOTE: please review and edit config if needed: $CFG_FILE"
fi


cfg() {
  # Usage: cfg <key> <default>
  local key="$1"
  local def="$2"
  python3 - "$CFG_FILE" "$key" "$def" <<'PY'
import json, sys
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
except Exception:
    print(default)
    raise SystemExit(0)

val = default
if isinstance(data, dict):
    val = data.get(key, default)

if val is None:
    print(default)
elif isinstance(val, bool):
    print("true" if val else "false")
else:
    s = str(val)
    print(s if s != "" else default)
PY
}


start_iter="$(cfg start_iter 0)"
max_iter="$(cfg max_iter 3)"

env_exploration_traj_dir="$(cfg env_exploration_traj_dir "")"
context_review_root_dir="$(cfg context_review_root_dir "")"
desktop_config_file="$(cfg desktop_config_file "./data/config_examples/desktop_configs_osworld_docker.json")"

seed="$(cfg seed 42)"
group_size="$(cfg group_size 8)"
mode="$(cfg mode visual)"
train_data_size="$(cfg train_data_size 500)"
val_data_size="$(cfg val_data_size 500)"
train_batch_size="$(cfg train_batch_size 16)"
val_batch_size="$(cfg val_batch_size 128)"

total_training_steps="$(cfg total_training_steps 75)"
save_freq="$(cfg save_freq 5)"
test_freq="$(cfg test_freq -1)"
n_gpus_per_node="$(cfg n_gpus_per_node 4)"
nnodes="$(cfg nnodes 2)"

learning_rate="$(cfg learning_rate 1e-6)"
ppo_mini_batch_size="$(cfg ppo_mini_batch_size 8192)"
ppo_micro_batch_size_per_gpu="$(cfg ppo_micro_batch_size_per_gpu 4)"
use_kl_loss="$(cfg use_kl_loss false)"
kl_loss_coef="$(cfg kl_loss_coef 0.0)"
kl_loss_type="$(cfg kl_loss_type low_var_kl)"
clip_ratio_low="$(cfg clip_ratio_low 0.2)"
clip_ratio_high="$(cfg clip_ratio_high 0.28)"
clip_ratio_c="$(cfg clip_ratio_c 10.0)"
loss_agg_mode="$(cfg loss_agg_mode token-mean)"
rollout_log_prob_micro_batch_size_per_gpu="$(cfg rollout_log_prob_micro_batch_size_per_gpu 4)"
limit_images="$(cfg limit_images 2)"
ref_log_prob_micro_batch_size_per_gpu="$(cfg ref_log_prob_micro_batch_size_per_gpu 4)"
use_kl_in_reward="$(cfg use_kl_in_reward false)"
kl_ctrl_kl_coef="$(cfg kl_ctrl_kl_coef 0.0)"
random_task_sampling="$(cfg random_task_sampling false)"

enable_cuajudge="$(cfg enable_cuajudge true)"
cuajudge_use_vllm_for_key_screenshot="$(cfg cuajudge_use_vllm_for_key_screenshot true)"
cuajudge_vllm_base_url="$(cfg cuajudge_vllm_base_url "http://IP:8000/v1")"
cuajudge_key_model="$(cfg cuajudge_key_model "YOUR_MODEL_NAME")"
cuajudge_outcome_model="$(cfg cuajudge_outcome_model gpt-5-mini)"
cuajudge_max_image="$(cfg cuajudge_max_image 50)"
cuajudge_score_threshold="$(cfg cuajudge_score_threshold 3)"

generated_tasks_root_dir="./data/tasks/examples/${SOFTWARE}/${TRAINING_MODEL_NAME}/${RUN_NAME}"
task_index_root_dir="./data/tasks/task_index/${SOFTWARE}/${TRAINING_MODEL_NAME}/${RUN_NAME}"

mkdir -p "$task_index_root_dir" "$generated_tasks_root_dir"

latest_global_step_dir() {
  local d="$1"
  local latest=""
  latest="$(ls -d "${d}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
  echo "$latest"
}

# Core loop
for ((iter=start_iter; iter<=max_iter; iter++)); do
  echo "============================================================"
  echo "[ACuRL] ITER=${iter}/${max_iter}"
  echo "============================================================"

  ITER_TASK_DIR="${generated_tasks_root_dir}/iter_${iter}"
  ITER_INDEX_DIR="${task_index_root_dir}/iter_${iter}"
  mkdir -p "$ITER_TASK_DIR" "$ITER_INDEX_DIR"

  # -----------------------
  # 1) Generate tasks
  # -----------------------
  echo "[ACuRL] >>> stage: generate_tasks (iter=${iter})"
  FEEDBACK_FILE=""
  if [[ "$iter" -gt 0 ]]; then
    FEEDBACK_FILE="${task_index_root_dir}/iter_$((iter-1))/performance_detailed.json"
    if [[ ! -f "$FEEDBACK_FILE" ]]; then
      echo "[ACuRL] ERROR: iter=${iter} requires feedback file: ${FEEDBACK_FILE}" >&2
      exit 1
    fi
  fi
  feedback_args=()
  if [[ -n "$FEEDBACK_FILE" ]]; then
    feedback_args+=(--feedback-file "$FEEDBACK_FILE")
  fi

  TASK_POOL_FILE="${ITER_INDEX_DIR}/${SOFTWARE}.generated_task_${EXPLORATION_MODEL}.json"

  # Skip task generation if the task pool file already exists (enables rerun-from-iter0 resume).
  diversity_args=()
  diversity_args+=(--enable-diversity true)
  if [[ "$iter" -eq 0 ]]; then
    effective_existing_tasks_dir="${ITER_TASK_DIR}"
  else
    effective_existing_tasks_dir="${generated_tasks_root_dir}/iter_$((iter-1))"
  fi

  [[ -s "$TASK_POOL_FILE" ]] || python3 "curriculum_task_generation/curriculum_task_generator.py" \
    --environment-exploration-dir "${env_exploration_traj_dir}" \
    --context-review-dir "${context_review_root_dir}" \
    --output-dir "$ITER_TASK_DIR" \
    --index-dir "$ITER_INDEX_DIR" \
    --api-key "${OPENAI_API_KEY}" \
    --model "$EXPLORATION_MODEL" \
    "${diversity_args[@]}" \
    --existing-tasks-dir "$effective_existing_tasks_dir" \
    --software "$SOFTWARE" \
    --iteration "$iter" \
    "${feedback_args[@]}"

  # -----------------------
  # 2) Prepare parquet (train/eval both need it)
  # -----------------------
  echo "[ACuRL] >>> stage: prepare_parquet (iter=${iter})"
  PARQUET_DIR="./data/tasks/parquet/${SOFTWARE}/${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}"
  mkdir -p "$PARQUET_DIR"
  # Skip parquet if both outputs exist (resume without re-materializing datasets).
  [[ -f "$PARQUET_DIR/train.parquet" && -f "$PARQUET_DIR/val.parquet" ]] || python3 "prepare_data.py" \
    --mode "$mode" \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}" \
    --data_dir "./data/tasks/examples" \
    --train_category "$SOFTWARE" \
    --val_category "$SOFTWARE" \
    --train_task_pool_file "$TASK_POOL_FILE" \
    --train_category_dir "$ITER_TASK_DIR" \
    --local_dir "$PARQUET_DIR" \
    --task_copy_times 1

  if [[ "$iter" -lt "$max_iter" ]]; then
    # Eval limits are required for iter < max_iter (we run eval to produce feedback for next iter).
    max_steps_eval_iter="$(cfg "max_steps_eval_iter_${iter}" "__MISSING__")"
    max_prompt_len_eval_iter="$(cfg "max_prompt_len_eval_iter_${iter}" "__MISSING__")"
    if [[ "$max_steps_eval_iter" == "__MISSING__" || "$max_prompt_len_eval_iter" == "__MISSING__" ]]; then
      echo "[ACuRL] ERROR: missing per-iter eval limits: max_steps_eval_iter_${iter} / max_prompt_len_eval_iter_${iter} (config: $CFG_FILE)" >&2
      exit 1
    fi
  fi

  # -----------------------
  # 3) Train (iter>0)
  # -----------------------
  if [[ "$iter" -gt 0 ]]; then
    echo "[ACuRL] >>> stage: train (iter=${iter})"
    max_steps_train_iter="$(cfg "max_steps_train_iter_${iter}" "__MISSING__")"
    max_prompt_len_iter="$(cfg "max_prompt_len_train_iter_${iter}" "__MISSING__")"
    if [[ "$max_steps_train_iter" == "__MISSING__" || "$max_prompt_len_iter" == "__MISSING__" ]]; then
      echo "[ACuRL] ERROR: missing per-iter train limits: max_steps_train_iter_${iter} / max_prompt_len_train_iter_${iter} (config: $CFG_FILE)" >&2
      exit 1
    fi

    CKPT_DIR="checkpoints/${PROJECT_NAME}/${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}"
    mkdir -p "$CKPT_DIR"
    TRAIN_RESUME_MODE="disable"
    TRAIN_RESUME_FROM="null"
    PREV_CKPT_DIR="checkpoints/${PROJECT_NAME}/${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_$((iter-1))"
    latest_curr="$(latest_global_step_dir "$CKPT_DIR")"
    if [[ -n "$latest_curr" ]]; then
      curr_step="$(basename "$latest_curr")"
      curr_step="${curr_step#global_step_}"
      # If current iter already reached total_training_steps, skip retraining this iter.
      if (( curr_step >= total_training_steps )); then
        TRAIN_RESUME_MODE="disable"
        TRAIN_RESUME_FROM="null"
      else
        # Resume within the same iter (framework-managed).
        TRAIN_RESUME_MODE="auto"
        TRAIN_RESUME_FROM="null"
      fi
    else
      latest_prev="$(latest_global_step_dir "$PREV_CKPT_DIR")"
      if [[ -n "$latest_prev" ]]; then
        # Start this iter from previous iter's latest checkpoint.
        TRAIN_RESUME_MODE="resume_path"
        TRAIN_RESUME_FROM="$latest_prev"
      fi
    fi

    # Global step override (clean iterative-training behavior):
    # - resume within the same iter: keep checkpoint step (null => no override)
    # - start a new iter from previous iter's checkpoint: reset counter to 0 for fresh logging/ckpt naming
    resume_global_step_override=null
    if [[ "$TRAIN_RESUME_MODE" == "resume_path" ]]; then
      resume_global_step_override=0
    fi

    # Skip submitting training job only when this iter is already complete (latest_curr exists and >= total_training_steps).
    [[ -n "$latest_curr" && "$TRAIN_RESUME_MODE" == "disable" ]] || ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="{\"env_vars\": {\"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\", \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\", \"WANDB_API_KEY\": \"${WANDB_API_KEY}\", \"ROCR_VISIBLE_DEVICES\": \"\", \"HYDRA_FULL_ERROR\": \"1\", \"PYTHONUNBUFFERED\": \"1\"}, \"excludes\": [\"**/.git/**\"]}" \
      -- python -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      +algorithm.filter_groups.enable=False \
      data.train_files="${PARQUET_DIR}/train.parquet" \
      data.val_files="${PARQUET_DIR}/val.parquet" \
      data.train_batch_size="$train_batch_size" \
      data.val_batch_size="$val_batch_size" \
      data.max_prompt_length="$max_prompt_len_iter" \
      data.max_response_length=512 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.trust_remote_code=True \
      data.return_raw_chat=True \
      data.dataloader_num_workers=0 \
      actor_rollout_ref.model.path="YOUR_MODEL_PATH" \
      actor_rollout_ref.actor.optim.lr="${learning_rate}" \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
      actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
      actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
      actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
      actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
      actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
      actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
      actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.enable_chunked_prefill=False \
      actor_rollout_ref.rollout.max_model_len="$max_prompt_len_iter" \
      actor_rollout_ref.rollout.max_num_batched_tokens="$max_prompt_len_iter" \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
      +actor_rollout_ref.rollout.limit_images="${limit_images}" \
      actor_rollout_ref.rollout.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=False \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      +actor_rollout_ref.actor.use_invalid_action_penalty=False \
      +actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
      algorithm.use_kl_in_reward="${use_kl_in_reward}" \
      algorithm.kl_ctrl.kl_coef="${kl_ctrl_kl_coef}" \
      +env.env_name=osworld \
      +env.desktop_config_file="$desktop_config_file" \
      +env.enable_cuajudge="$enable_cuajudge" \
      +env.enable_rule_based=true \
      +env.cuajudge_key_model="$cuajudge_key_model" \
      +env.cuajudge_outcome_model="$cuajudge_outcome_model" \
      +env.cuajudge_max_image="$cuajudge_max_image" \
      +env.cuajudge_score_threshold="$cuajudge_score_threshold" \
      +env.use_vllm_for_key_screenshot="$cuajudge_use_vllm_for_key_screenshot" \
      +env.vllm_base_url="$cuajudge_vllm_base_url" \
      +env.seed="$seed" \
      +env.max_steps="$max_steps_train_iter" \
      +env.rollout.n="$group_size" \
      +env.is_sample_mode=false \
      +env.random_task_sampling="$random_task_sampling" \
      +env.image_save_dir="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}" \
      +env.train_subfolder="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}" \
      trainer.critic_warmup=0 \
      trainer.logger=['console','wandb'] \
      trainer.project_name="$PROJECT_NAME" \
      trainer.experiment_name="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}" \
      trainer.default_local_dir="${CKPT_DIR}" \
      trainer.resume_mode="$TRAIN_RESUME_MODE" \
      trainer.resume_from_path="$TRAIN_RESUME_FROM" \
      trainer.resume_global_step=${resume_global_step_override} \
      trainer.n_gpus_per_node="$n_gpus_per_node" \
      trainer.nnodes="$nnodes" \
      trainer.save_freq="$save_freq" \
      trainer.test_freq="$test_freq" \
      trainer.val_only=false \
      trainer.val_before_train=false \
      trainer.total_training_steps="$total_training_steps"
  fi

  if [[ "$iter" -lt "$max_iter" ]]; then
    # -----------------------
    # 4) Eval task quality (sampled rollouts)
    # -----------------------
    echo "[ACuRL] >>> stage: eval_task_quality (iter=${iter})"
    EVAL_IMAGE_SAVE_DIR="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}/task_quality_eval"
    RESULTS_DIR="trajectory/${EVAL_IMAGE_SAVE_DIR}/test"
    RESUME_MODE="disable"
    RESUME_FROM="null"
    if [[ "$iter" -gt 0 ]]; then
      CKPT_DIR="checkpoints/${PROJECT_NAME}/${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}"
      latest="$(latest_global_step_dir "$CKPT_DIR")"
      if [[ -n "$latest" ]]; then
        RESUME_MODE="resume_path"
        RESUME_FROM="$latest"
      fi
    fi

    # Skip eval if result dir already exists (resume without rerunning rollouts).
    [[ -d "$RESULTS_DIR" ]] || ray job submit --address="http://127.0.0.1:8265" \
      --runtime-env-json="{\"env_vars\": {\"OPENAI_API_KEY\": \"${OPENAI_API_KEY}\", \"OPENAI_BASE_URL\": \"${OPENAI_BASE_URL}\", \"WANDB_API_KEY\": \"${WANDB_API_KEY}\", \"ROCR_VISIBLE_DEVICES\": \"\", \"HYDRA_FULL_ERROR\": \"1\", \"PYTHONUNBUFFERED\": \"1\"}, \"excludes\": [\"**/.git/**\"]}" \
      -- python -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      +algorithm.filter_groups.enable=False \
      data.train_files="${PARQUET_DIR}/train.parquet" \
      data.val_files="${PARQUET_DIR}/train.parquet" \
      data.train_batch_size="$train_batch_size" \
      data.val_batch_size="$train_batch_size" \
      data.max_prompt_length="$max_prompt_len_eval_iter" \
      data.max_response_length=512 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.trust_remote_code=True \
      data.return_raw_chat=true \
      data.dataloader_num_workers=0 \
      actor_rollout_ref.model.path="YOUR_MODEL_PATH" \
      actor_rollout_ref.actor.optim.lr="${learning_rate}" \
      actor_rollout_ref.model.use_remove_padding=true \
      actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
      actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
      actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
      actor_rollout_ref.actor.kl_loss_type="${kl_loss_type}" \
      actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
      actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
      actor_rollout_ref.actor.clip_ratio_c="${clip_ratio_c}" \
      actor_rollout_ref.actor.loss_agg_mode="${loss_agg_mode}" \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.enable_chunked_prefill=false \
      actor_rollout_ref.rollout.max_model_len="$max_prompt_len_eval_iter" \
      actor_rollout_ref.rollout.max_num_batched_tokens="$max_prompt_len_eval_iter" \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${rollout_log_prob_micro_batch_size_per_gpu}" \
      actor_rollout_ref.rollout.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ref_log_prob_micro_batch_size_per_gpu}" \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      +actor_rollout_ref.actor.use_invalid_action_penalty=False \
      +actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
      algorithm.use_kl_in_reward="${use_kl_in_reward}" \
      algorithm.kl_ctrl.kl_coef="${kl_ctrl_kl_coef}" \
      +env.env_name=osworld \
      +env.enable_cuajudge="$enable_cuajudge" \
      +env.enable_rule_based=false \
      +env.cuajudge_key_model="$cuajudge_key_model" \
      +env.cuajudge_outcome_model="$cuajudge_outcome_model" \
      +env.cuajudge_max_image="$cuajudge_max_image" \
      +env.cuajudge_score_threshold="$cuajudge_score_threshold" \
      +env.use_vllm_for_key_screenshot="$cuajudge_use_vllm_for_key_screenshot" \
      +env.vllm_base_url="$cuajudge_vllm_base_url" \
      +env.desktop_config_file="$desktop_config_file" \
      +env.seed="$seed" \
      +env.max_steps="$max_steps_eval_iter" \
      +env.rollout.n="$group_size" \
      +env.is_sample_mode=true \
      +env.image_save_dir="${EVAL_IMAGE_SAVE_DIR}" \
      +env.val_subfolder="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}" \
      trainer.critic_warmup=0 \
      trainer.logger=['console'] \
      trainer.project_name="$PROJECT_NAME" \
      trainer.experiment_name="${TRAINING_MODEL_NAME}/${RUN_NAME}/iter_${iter}/task_quality_eval" \
      trainer.resume_mode="${RESUME_MODE}" \
      trainer.resume_from_path="${RESUME_FROM}" \
      trainer.n_gpus_per_node="$n_gpus_per_node" \
      trainer.nnodes="$nnodes" \
      trainer.save_freq=-1 \
      trainer.test_freq=-1 \
      trainer.val_only=true \
      trainer.val_before_train=true \
      trainer.total_training_steps=1

    # -----------------------
    # 5) Calculate performance (canonical implementation)
    # -----------------------
    echo "[ACuRL] >>> stage: calculate_performance (iter=${iter})"
    PERF_BASE="${ITER_INDEX_DIR}/performance_detailed.json"
    # Skip performance calc if output already exists (resume without recomputing).
    [[ -f "$PERF_BASE" ]] || python3 "curriculum_task_generation/calculate_performance.py" \
      --results_dir "$RESULTS_DIR" \
      --task_pool_file "$TASK_POOL_FILE" \
      --output_file "$PERF_BASE"
  fi

  echo "[ACuRL] iter ${iter} done."
done

echo "[ACuRL] all done."
