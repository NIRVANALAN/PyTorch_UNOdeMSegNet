source source.sh
#python3 -u bin.py --config cfgs/unet.yaml
python3 -u train/wnet_trainer.py --config cfgs/unet_smp_512.yaml

