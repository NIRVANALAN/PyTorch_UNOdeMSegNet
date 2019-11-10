partition=${1}
config=${2}
num_gpus=1
srun --mpi=pmi2 -p ${partition} \
  -n${num_gpus} --gres=gpu:1 --ntasks-per-node=1 \
  --job-name=UT python3 -u bin/inference.py --config cfgs/unet_smp_inference.yaml
#python3 -u bin.py --config cfgs/unet.yaml
#python3 -u train/inference.py --config cfgs/unet_smp_inference.yaml

