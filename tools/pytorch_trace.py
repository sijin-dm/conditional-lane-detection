import argparse
import io

import mmcv
import torch
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet pytorch model conversion to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out', type=str, required=True, help='output ONNX filename')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 800],
        help='input image size')
    parser.add_argument(
        '--passes', type=str, nargs='+', help='ONNX optimization passes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    # Only support CPU mode for now
    model.cpu().eval()
    # Customized ops are not supported, use torchvision ops instead.
    # for m in model.modules():
    #     if isinstance(m, (RoIPool, RoIAlign)):
    #         # set use_torchvision on-the-fly
    #         m.use_torchvision = True

    torch.jit.save(model, 'condlanenet_traced.zip')






if __name__ == '__main__':
    main()
