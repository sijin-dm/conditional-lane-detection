import os
import numpy as np
import torch
import torch.nn as nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module
class CurvelanesRnnTrace(nn.Module):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 loss_weights={},
                 num_classes=1):
        super(CurvelanesRnnTrace, self).__init__()
        self.num_classes = num_classes
        self.head = head
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.bbox_head = build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        # super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

    def forward(self, img):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        output = self.backbone(img)
        output, _ = self.neck(output)
        output= self.bbox_head.forward_test(output, 0.5)
        return list(output)