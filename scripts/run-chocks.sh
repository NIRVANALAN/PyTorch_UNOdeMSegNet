module load cuda/10.1

# chocks env
source /work/04776/chocks/stampede2/scripts/conda3.sh
conda activate mseg
python3 -u train/smp_train.py --config cfgs/unet_smp_64.yaml

