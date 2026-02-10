export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export ROCR_VISIBLE_DEVICES=""
export OPENAI_BASE_URL="https://api.openai.com/v1"


SOFTWARE=${1:-"libreoffice_impress"}

##### model parameters #####
Model="YOUR_MODEL_PATH"
Model_name="YOUR_MODEL_NAME"


##### configuration parameters #####
DESKTOP_CONFIG_FILE="./data/config_examples/environment_config.json"
mkdir -p "./logs/${Model_name}/context_review"
LOG_FILE="./logs/${Model_name}/context_review/${SOFTWARE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

##### training parameters #####
train_data_size=500
val_data_size=500
train_batch_size=8
val_batch_size=8
group_size=8
ppo_mini_batch_size_base=512
ppo_mini_batch_size=$((ppo_mini_batch_size_base * group_size))
save_freq=-1
test_freq=-1
total_training_steps=100
max_steps=20
length_limit=15000
seed=42

mode=visual
sample_mode=False
random_task_sampling=False

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"

cuajudge_key_model="gpt-5-mini"
cuajudge_outcome_model="gpt-5-mini"
cuajudge_max_image=50
cuajudge_score_threshold=3

##### data parameters #####
category="$SOFTWARE"
data_dir="./data/tasks/examples"
train_task_pool_file_name="${SOFTWARE}.context_review.json"
train_task_pool_file="./data/tasks/task_index/${train_task_pool_file_name}"

python3 ./prepare_data.py \
    --mode ${mode} \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --data_dir $data_dir \
    --train_category $category \
    --val_category $category \
    --train_task_pool_file $train_task_pool_file \
    --local_dir ./data/tasks/parquet/${Model_name}/context_review/${SOFTWARE} \
    --task_copy_times 1

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="{\"env_vars\": {\"OPENAI_API_KEY\": \"$OPENAI_API_KEY\", \"HYDRA_FULL_ERROR\": \"1\", \"PYTHONUNBUFFERED\": \"1\", \"ROCR_VISIBLE_DEVICES\": \"\", \"PYTHONFAULTHANDLER\": \"1\"}, \"excludes\": [\"**/.git/**\"]}" \
    -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.filter_groups.enable=False \
    data.train_files=./data/tasks/parquet/${Model_name}/context_review/${SOFTWARE}/train.parquet \
    data.val_files=./data/tasks/parquet/${Model_name}/context_review/${SOFTWARE}/train.parquet \
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
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.max_model_len=$length_limit \
    actor_rollout_ref.rollout.max_num_batched_tokens=$length_limit \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
    +env.enable_cuajudge=false \
    +env.enable_rule_based=true \
    +env.cuajudge_key_model=$cuajudge_key_model \
    +env.cuajudge_outcome_model=$cuajudge_outcome_model \
    +env.cuajudge_max_image=$cuajudge_max_image \
    +env.cuajudge_score_threshold=$cuajudge_score_threshold \
    +env.desktop_config_file=$DESKTOP_CONFIG_FILE \
    +env.seed=$seed \
    +env.max_steps=$max_steps \
    +env.rollout.n=$group_size \
    +env.is_sample_mode=$sample_mode \
    +env.image_save_dir=${Model_name}/context_review/${SOFTWARE} \
    +env.train_subfolder=context_review \
    +env.val_subfolder=context_review \
    trainer.resume_mode=auto \
    trainer.resume_from_path=null \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ACuRL' \
    trainer.experiment_name=${Model_name}/context_review/${SOFTWARE} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.total_training_steps=$total_training_steps