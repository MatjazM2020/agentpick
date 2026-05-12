#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --job-name=agentpick_data
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --output=/d/hpc/home/mm11484/agentpick_data/agentpick_data.log
#SBATCH --time=20:00:00

# Set working directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p logs data

# Set up environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
export PYTHONPATH=/d/hpc/home/mm11484/agentpick_data/src:$PYTHONPATH

# CUDA safety environment variables for robust GPU memory handling
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Install dependencies
pip install -q -r /d/hpc/home/mm11484/agentpick_data/requirements.txt

echo "======================================"
echo "SLURM Job Information"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Time Limit: $SLURM_TIME_LIMIT"
echo "Working Directory: $PWD"
echo "Node: $(hostname)"
echo "======================================"
echo ""

# Run vectorization
python -m hf_vectorizer.vectorizer \
    --data-dir ./data \
    --embedding-model "BAAI/bge-large-en-v1.5" \
    --batch-size 8

echo ""
echo "Job completed!"
