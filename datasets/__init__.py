import torch.utils.data
import torchvision

from .high import build as build_high

data_path = {
    'SHA': './data/ShanghaiTech/part_A/',
    'HIGH': r'D:\kk\PET\data\high_2048',
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'HIGH':
        return build_high(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
