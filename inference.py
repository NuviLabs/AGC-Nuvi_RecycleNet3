#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


@author: Nuvilabs - Luca Medeiros, luca.medeiros@nuvi-labs.com
"""

# from utils.mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2
import torch

from mmcv import Config
from mmdet.datasets import pipelines
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter


class Nuvi_RecycleNet():
    def __init__(self,
                 config_file,
                 checkpoint_file,
                 threshold,
                 device='cuda:0',
                 tta=True):

        self.config_file = config_file
        self.config_drs = Config.fromfile(config_file)
        self.checkpoint_file = checkpoint_file
        self.threshold = threshold  # Only boxes with the score larger than this will be detected
        self.tta = tta # Perform TTA on not detected images

        # Build the model from a config file and a checkpoint file
        # self.model = init_detector(self.config_file, self.checkpoint_file, device=device)
        self.model = torch.load(self.checkpoint_file)
        self.model.eval()
        self.model.cuda()


        print('Model loaded sucessfully. Ready to perform inference.')
        print('Threshold: ', self.threshold)
        self.classes = {0: 'paper',
                        1: 'paper_pack',
                        2: 'steel',
                        3: 'glass',
                        4: 'PET',
                        5: 'plastic',
                        6: 'plasticbag'
                        }

    def augmentIMG(self, img_array, augment_type):
        if augment_type == 'LR':
            print('augmented', augment_type)
            img = np.fliplr(img_array)

        elif augment_type == 'UDR':
            print('augmented', augment_type)
            img = np.flipud(img_array)

        elif augment_type == 'RT':
            print('augmented', augment_type)
            img = np.rot90(img_array)

        return imggit 
        
    def infer_drs(self, img):
        """Inference image(s) with the detector.
    
        Args:
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.
    
        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
        """
        cfg = self.config_drs
        device = next(self.model.parameters()).device  # model device
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
    
