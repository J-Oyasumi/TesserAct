#!/bin/bash
#SBATCH --job-name=train_tesseract_pointmap
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dgx-b200
#SBATCH --mem=256GB
#SBATCH --time=72:00:00
#SBATCH --output=./logs/train_tesseract_pointmap/%x-%j.out
#SBATCH --error=./logs/train_tesseract_pointmap/%x-%j.err

cd $HOME/workspace/TesserAct
eval "$(conda shell.bash hook)"
conda activate ta

export PYTHONPATH=$PYTHONPATH:$HOME/workspace/TesserAct
echo "Host: $(hostname)"
echo "Current Python: $(which python)"

export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="online"
# export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# Add NCCL debug and optimization settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=1800
export NCCL_SOCKET_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i 20000-60000 -n 1)
MASTER_ADDR=$HOSTNAME
MASTER_PORT=$SLURM_JOB_ID
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

# Training Configurations
# Experiment with as many hyperparameters as you want!
LEARNING_RATES=("5e-5")
LR_SCHEDULES=("constant_with_warmup")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("1000")
frame_length=49

VALIDATION_PROMPT="\"put clothes in laundry machine Trossen WidowX 250 robot arm:::"\
"put clothes in laundry machine Trossen WidowX 250 robot arm:::"\
"put clothes in laundry machine Trossen WidowX 250 robot arm:::"\
"put clothes in laundry machine Trossen WidowX 250 robot arm"\
"\""

VALIDATION_IMAGES="data/bridgev2/processed/1/video/rgb.mp4:::"\
"data/bridgev2/processed/2/video/rgb.mp4:::"\
"data/bridgev2/processed/3/video/rgb.mp4:::"\
"data/bridgev2/processed/4/video/rgb.mp4"


DATA_ROOT="data"
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

for learning_rate in "${LEARNING_RATES[@]}"; do
  for lr_schedule in "${LR_SCHEDULES[@]}"; do
    for optimizer in "${OPTIMIZERS[@]}"; do
      for steps in "${MAX_TRAIN_STEPS[@]}"; do
        output_dir="./runs/models/tesseract-pointmap-sft-full__framelength_${frame_length}__optimizer_${optimizer}__lr-schedule_${lr_schedule}__learning-rate_${learning_rate}/"
        PY_BIN_DIR=$(dirname $(which python))
        launcher_cmd="python -m torch.distributed.run --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
          --rdzv_backend static --nnodes $NNODES --nproc_per_node=$NUM_GPUS --rdzv_id=$NODE_RANK"
        echo "Using launcher command: $launcher_cmd"

        cmd="$launcher_cmd \
          tesseract/i2v_pointmap_sft.py \
          --pretrained_model_name_or_path $MODEL_PATH \
          --dataset_file cache/samples_pointmap.json \
          --data_root $DATA_ROOT \
          --height_buckets 240 256 480 512 720 \
          --width_buckets 320 512 640 854 1280 \
          --frame_buckets 9 17 25 33 49 \
          --dataloader_num_workers 2 \
          --pin_memory \
          --validation_prompt $VALIDATION_PROMPT \
          --validation_images $VALIDATION_IMAGES \
          --validation_prompt_separator ::: \
          --num_validation_videos 1 \
          --validation_steps 100 \
          --seed 42 \
          --mixed_precision bf16 \
          --output_dir $output_dir \
          --height 480 \
          --width 640 \
          --guidance_scale 7.5 \
          --max_num_frames $frame_length \
          --train_batch_size 2 \
          --max_train_steps $steps \
          --checkpointing_steps 100 \
          --checkpoints_total_limit 15 \
          --resume_from_checkpoint latest \
          --gradient_accumulation_steps 1 \
          --learning_rate $learning_rate \
          --lr_scheduler $lr_schedule \
          --lr_warmup_steps 200 \
          --lr_num_cycles 1 \
          --noised_image_dropout 0.05 \
          --gradient_checkpointing \
          --enable_slicing \
          --enable_tiling \
          --optimizer $optimizer \
          --beta1 0.9 \
          --beta2 0.95 \
          --weight_decay 0.001 \
          --max_grad_norm 1.0 \
          --allow_tf32 \
          --report_to wandb \
          --ignore_learned_positional_embeddings \
          --nccl_timeout 1800"

        echo "Running command: $cmd"
        eval $cmd
        echo -ne "-------------------- Finished executing script --------------------\n\n"
      done
    done
  done
done
