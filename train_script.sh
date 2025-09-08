#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your_username>

WORK_DIR="/vol/bitbucket/kst24/my_instant_policy"
CUDA_VERSION="11.8.0"
ENV_NAME="myenv"
echo "START"
export PENV=/vol/bitbucket/${USER}/${ENV_NAME}

export PATH=/vol/bitbucket/kst24/fyp/my_instant_policy/:$PATH
source "/vol/cuda/${CUDA_VERSION}/setup.sh"



# Install required packages (if not already installed)
# pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install --user pygame numpy matplotlib

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
# python3 -m virtualenv $PENV
source $PENV/bin/activate
pip install -r "${WORK_DIR}/requirements.txt"

cp ./checkpoints/ckpt_0010000.pth ./agent.pth

python3 -u "${WORK_DIR}/train.py"
