set -e

# conda init bash
conda activate Pytorch-gpu
python image_matting.py
conda deactivate
echo "Finished"