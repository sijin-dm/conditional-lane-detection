import argparse
import os
import numpy as np
import copy
import cv2
import mmcv
import torch
import torch.distributed as dist
import PIL.Image
import PIL.ImageDraw
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
#from mmdet.utils.general_utils import mkdir
from tools.condlanenet.post_process import CurvelanesPostProcessor
from tools.condlanenet.lane_metric import LaneMetricCore
from tools.condlanenet.common import convert_coords_formal, parse_anno, COLORS


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='seg checkpoint file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--hm_thr', type=float, default=0.5)
    parser.add_argument('--show', action='store_true')
    parser.add_argument(
        '--show_dst',
        default=None,
        help='path to save visualized results.')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='whether to compare pred and gt')
    parser.add_argument('--eval_width', type=float, default=224)
    parser.add_argument('--eval_height', type=float, default=224)
    parser.add_argument('--lane_width', type=float, default=5)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def adjust_result(lanes,
                  crop_offset,
                  crop_shape,
                  img_shape,
                  tgt_shape=(590, 1640)):
    h_img, w_img = img_shape[:2]
    ratio_x = crop_shape[1] / w_img
    ratio_y = crop_shape[0] / h_img
    offset_x, offset_y = crop_offset

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(tuple(pt))
            if len(pts) > 1:
                results.append(pts)
    return results

def vis_one_for_paper(results,
                      filename,
                      ori_shape,
                      lane_width=11,
                      draw_gt=True):
    img = cv2.imread(filename)
    img_ori = copy.deepcopy(img)
    img_gt = copy.deepcopy(img)
    img_pil = PIL.Image.fromarray(img)
    img_gt_pil = PIL.Image.fromarray(img_gt)
    for idx, pred_lane in enumerate(results):
        PIL.ImageDraw.Draw(img_pil).line(
            xy=pred_lane, fill=COLORS[idx + 1], width=lane_width)

    img = np.array(img_pil, dtype=np.uint8)
    img_gt = np.array(img_gt_pil, dtype=np.uint8)
    return img, img_gt, img_ori


def single_gpu_test(seg_model,
                    data_loader,
                    show=None,
                    hm_thr=0.3,
                    evaluate=True,
                    eval_width=224,
                    eval_height=224,
                    lane_width=5,
                    mask_size=(1, 40, 100)):
    seg_model.eval()
    dataset = data_loader.dataset
    post_processor = CurvelanesPostProcessor(
        mask_size=mask_size, hm_thr=hm_thr)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            filename = data['img_metas'].data[0][0]['filename']
            sub_name = data['img_metas'].data[0][0]['sub_img_name']
            img_shape = data['img_metas'].data[0][0]['img_shape']
            ori_shape = data['img_metas'].data[0][0]['ori_shape']
            crop_offset = data['img_metas'].data[0][0]['crop_offset']
            crop_shape = data['img_metas'].data[0][0]['crop_shape']
            
            seeds, hm = seg_model(data['img'])
            downscale = data['img_metas'].data[0][0]['down_scale']
            lanes, seeds = post_processor(seeds, downscale)

            # This is the predicted points of each lane.
            result = adjust_result(
                lanes=lanes,
                crop_offset=crop_offset,
                crop_shape=crop_shape,
                img_shape=img_shape,
                tgt_shape=ori_shape)

        if show is not None:
            filename = data['img_metas'].data[0][0]['filename']
            img_vis, _, _ = vis_one_for_paper(
                result,
                filename,
                ori_shape=ori_shape,
                draw_gt=False,
                lane_width=13)
            basename = os.path.basename(sub_name)
            dst_show_dir = os.path.join(show, basename)
            cv2.imwrite(dst_show_dir, img_vis)

        batch_size = data['img'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    show_dst = args.show_dst
    if show_dst is not None:
        print("Saving output to: {}".format(show_dst))
        os.makedirs(show_dst, exist_ok=True)

    single_gpu_test(
        seg_model=model,
        data_loader=data_loader,
        show=show_dst,
        hm_thr=args.hm_thr,
        evaluate=args.evaluate,
        eval_width=args.eval_width,
        eval_height=args.eval_height,
        lane_width=args.lane_width,
        mask_size=cfg.mask_size)


if __name__ == '__main__':
    main()
