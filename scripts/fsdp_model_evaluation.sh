export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export ROCR_VISIBLE_DEVICES=""
export OPENAI_BASE_URL="https://api.openai.com/v1"

SOFTWARE="${1:-libreoffice_impress}"
GENERATOR_MODEL="${2:-gpt-5}"
CHECKPOINT_DIR="${3:-}"
GLOBAL_STEP="${4:-}"

##### configuration parameters #####
DESKTOP_CONFIG_FILE="./data/config_examples/environment_exploration.json"
mkdir -p "./logs/${Model_name}/evaluation"
LOG_FILE="./logs/${Model_name}/evaluation/${SOFTWARE}_${GENERATOR_MODEL}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

##### model parameters #####
Model="BASE_MODEL_PATH"
Model_name="BASE_MODEL_NAME"

##### training/eval parameters #####
train_data_size=500
val_data_size=500
train_batch_size=16
val_batch_size=128
group_size=8
ppo_mini_batch_size_base=512
ppo_mini_batch_size=$((ppo_mini_batch_size_base * group_size))
save_freq=-1
test_freq=-1
total_training_steps=1
max_steps=25
length_limit=19000

mode=visual

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
kl_loss_type=low_var_kl
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
learning_rate=1e-6
clip_ratio_c=10.0

seed=42

##### data parameters #####
data_dir="./data/tasks/examples"

train_task_pool_file_name="${SOFTWARE}.environment_exploration.json"
train_task_pool_file="./data/tasks/task_index/${train_task_pool_file_name}"

parquet_dir="./data/tasks/parquet/${Model_name}/evaluation/${SOFTWARE}"

val_categories=("libreoffice_calc" "libreoffice_writer" "libreoffice_impress" "ow-xlsx" "ow-docx" "ow-pptx" "thunderbird" "KAlgebra" "Celestia")

python3 ./prepare_data.py \
  --mode ${mode} \
  --train_data_size "$train_data_size" \
  --val_data_size "$val_data_size" \
  --data_dir "$data_dir" \
  --train_category "$SOFTWARE" \
  --val_category "${val_categories[@]}" \
  --train_task_pool_file "$train_task_pool_file" \
  --local_dir "$parquet_dir" \
  --task_copy_times 1

##### checkpoint parameters #####
##### evaluation #####
checkpoint_path="$CHECKPOINT_DIR"
if [ -n "$GLOBAL_STEP" ]; then
  checkpoint_path="$CHECKPOINT_DIR/global_step_${GLOBAL_STEP}"
fi

global_step_name="$(basename "$checkpoint_path")"

echo "Evaluating checkpoint: $global_step_name"

image_save_dir="${Model_name}/evaluation/${SOFTWARE}_${GENERATOR_MODEL}/${global_step_name}"
eval_experiment_name="${Model_name}/evaluation/${SOFTWARE}_${GENERATOR_MODEL}/${global_step_name}"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"env_vars\": {\"OPENAI_API_KEY\": \"$OPENAI_API_KEY\", \"OPENAI_BASE_URL\": \"$OPENAI_BASE_URL\", \"WANDB_API_KEY\": \"$WANDB_API_KEY\", \"HYDRA_FULL_ERROR\": \"1\", \"VLLM_BATCH_INVARIANT\": \"1\", \"PYTHONUNBUFFERED\": \"1\", \"ROCR_VISIBLE_DEVICES\": \"$ROCR_VISIBLE_DEVICES\", \"PYTHONFAULTHANDLER\": \"1\"}, \"excludes\": [\"**/.git/**\"]}" \
    -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.filter_groups.enable=False \
    data.train_files="$parquet_dir/val.parquet" \
    data.val_files="$parquet_dir/val.parquet" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$length_limit \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.trust_remote_code=True \
    data.return_raw_chat=True \
    data.dataloader_num_workers=0 \
    actor_rollout_ref.model.path=${Model} \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=$clip_ratio_c \
    actor_rollout_ref.actor.loss_agg_mode=$loss_agg_mode \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=$length_limit \
    actor_rollout_ref.rollout.max_num_batched_tokens=$length_limit \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.actor.use_invalid_action_penalty=False \
    +actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    +env.env_name=osworld \
    +env.desktop_config_file=$DESKTOP_CONFIG_FILE \
    +env.enable_cuajudge=false \
    +env.enable_rule_based=true \
    +env.seed=42 \
    +env.max_steps=$max_steps \
    +env.rollout.n=$group_size \
    +env.is_sample_mode=false \
    +env.random_task_sampling=false \
    +env.image_save_dir=$image_save_dir \
    trainer.resume_mode="resume_path" \
    trainer.resume_from_path="$checkpoint_path" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='CUA_RL' \
    trainer.experiment_name="$eval_experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=2 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.total_training_steps=$total_training_steps


