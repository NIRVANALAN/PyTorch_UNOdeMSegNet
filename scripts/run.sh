#source source.sh
partition=${1}
num_gpus=1
srun --mpi=pmi2 -p ${partition} \
  -n${num_gpus} --gres=gpu:1 --ntasks-per-node=1 \
  --job-name=UT python3 -u bin/train.py --config cfgs/unet_smp_512_odesegnet.yaml
#python3 -u train/smp_train.py --config cfgs/unet_smp_768_weight.yaml

