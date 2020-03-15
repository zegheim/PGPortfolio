#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1 # GPUs requested
#SBATCH --cpus-per-task=1 # CPUs requested
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-02:00:00

export CUDA_HOME=/opt/cuda-10.0.130

export CUDNN_HOME=/opt/cuDNN-7.6.0.64_9.2

export STUDENT_ID=\$(whoami)

export LD_LIBRARY_PATH=\${CUDNN_HOME}/lib64:\${CUDA_HOME}/lib64:\$LD_LIBRARY_PATH

export LIBRARY_PATH=\${CUDNN_HOME}/lib64:\$LIBRARY_PATH

export CPATH=\${CUDNN_HOME}/include:\$CPATH

export PATH=\${CUDA_HOME}/bin:\${PATH}

export PYTHON_PATH=\$PATH

mkdir -p /disk/scratch/\${STUDENT_ID}


export TMP=/disk/scratch/\${STUDENT_ID}

export JOB_ID=\${SLURM_JOB_NAME%???}

mkdir -p \${TMP}/datasets
export DATASET_DIR=\${TMP}/datasets/\${JOB_ID}

export OUTPUT_DIR=/home/\${STUDENT_ID}/Experiments/FinalHyperparameterTuning/\${JOB_ID}
mkdir -p \${OUTPUT_DIR}

date
echo \"Copying data..\"

rsync -uap --progress /home/\${STUDENT_ID}/PGPortfolio \${DATASET_DIR}

date
echo \"Finished copying data, starting training\"

# Activate the relevant virtual environment:

source /home/\${STUDENT_ID}/miniconda3/bin/activate pgp

cd \${DATASET_DIR}/PGPortfolio

# Start experiment

python -m scripts.json_to_cli "$@"

python -m main --mode=generate --repeat=1
python -m main --mode=train --processes=1 --device=gpu --restore_dir="pretrained/netfile"

echo \"Copying results to main node\"
rsync -uap --progress train_package/ \${OUTPUT_DIR}

date
echo \"Finished\"
