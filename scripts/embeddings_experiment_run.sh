#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --partition=Teach-Short
#SBATCH --gres=gpu:1 # GPUs requested
#SBATCH --cpus-per-task=1 # CPUs requested
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-01:00:00

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

mkdir -p \${TMP}/datasets
export DATASET_DIR=\${TMP}/datasets

export JOB_ID=\${SLURM_JOB_NAME%???}

export TEMP_OUTPUT_DIR=\${TMP}/\${JOB_ID}
mkdir -p \${TEMP_OUTPUT_DIR}

export OUTPUT_DIR=/home/\${STUDENT_ID}/PGPortfolio/HyperparameterTuning/\${JOB_ID}
mkdir -p \${OUTPUT_DIR}

date
echo \"Copying data..\"

rsync -uap --progress /home/\${STUDENT_ID}/\${DATASET_DIR}

date
echo \"Finished copying data, starting training\"

# Activate the relevant virtual environment:

source /home/\${STUDENT_ID}/miniconda3/bin/activate pgp

cd \${DATASET_DIR}

# Start experiment

python -m scripts.json_to_cli \
"$@"

python

echo \"Copying results to main node\"
rsync -uap --progress \${TEMP_OUTPUT_DIR}/ \${OUTPUT_DIR}

date
echo \"Finished\"
