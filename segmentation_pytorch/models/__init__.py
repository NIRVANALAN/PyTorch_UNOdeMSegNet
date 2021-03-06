from .ConcatSquash2D import *
from .mnodeunet import *
from .wavelet_model import *
import segmentation_pytorch as smp
from .unet import NestedUNet, UNet, ResUNet
import pdb

FACTORY = {
    'resunet': ResUNet,
    'unet': UNet,
    'unet_ode': UNOdeMSegNet,
    'unetplus': NestedUNet,
    # 'unet_wavelet': WaveletModel
}


def create_model(args):
    num_classes = args.data.get('num_classes', 8)
    arch = args.model.arch
    print(f'creating model: {args.model}')
    if 'ode' in arch:
        model = UNOdeMSegNet(dim_in=args.model.dim_in,
                             dim_latent=args.model.dim_latent,
                             num_res=args.model.num_res,
                             scale_factor=args.model.scale_factor,
                             n_classes=num_classes)
        # pdb.set_trace()
    else:
        model = FACTORY[arch](encoder_name=args.model.encoder,
                              encoder_weights=None,
                              activation=None,
                              classes=num_classes)
    return model
    # if args.data.get('wavelet', False):
    # 	model = WaveletModel(
    # 		encoder_name=args.model.arch,
    # 		# encoder_weights='imagenet',
    # 		encoder_weights=None,
    # 		activation='softmax',  # whatever, will do .max during metrics calculation
    # 		classes=num_classes)
    # else:
    # 	# model = deeplabv3_resnet50(num_classes=num_classes, pretrained=True)
    #
    # 	model = smp.Unet(
    # 		encoder_name=arch,
    # 		# encoder_weights='imagenet',
    # 		encoder_weights=None,
    # 		activation='softmax',  # whatever, will do .max during metrics calculation
    # 		classes=num_classes, multi_stage=multi_stage)
    # 	pass
    # return model(arch=name, *args, **kwargs)
