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

        return img
    
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
        # build the data pipeline
        test_pipeline = pipelines.Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [0])[0]
        else:
            for m in self.model.modules():
                assert not isinstance(
                    m, RoIPool
                ), 'CPU inference with RoIPool is not supported currently.'
            # just get the actual data from DataContainer
            data['img_metas'] = data['img_metas'][0].data
    
        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)[0]
        return result


    def predict(self, img_path):
        imageArray = cv2.imread(img_path)
        # Run inference using a model on a single picture -> img can be either path or array
        result = self.infer_drs(imageArray)
        # result = inference_detector(self.model, imageArray)
        res_idxs = [[i, k[0]] for i, k in enumerate(result) if k.size != 0 and (k[:,4] > self.threshold).any()]
        if self.tta and not res_idxs:
            # Determine which augmentations to do when there is no detection. In order
            aug_type = ['LR', 'UDR', 'RT', 'Break']
            aug_idx = 0
            res_idxs = []
            while not res_idxs:
                augment_type = aug_type[aug_idx]
                if augment_type == 'Break':
                    result = [np.asarray([]) for i in range(8)]
                    break
                print('TTA type: ', augment_type)
                img_transformed = self.augmentIMG(imageArray, augment_type)
                result = self.infer_drs(imageArray)
                res_idxs = [[i, k[0]] for i, k in enumerate(result[0]) if k.size != 0 and (k[:,4] > self.threshold).any()]

        json_result = self.make_json(res_idxs)

        return json_result

    def make_json(self, results):
        json_dict = {'Annotations': []}
        for result in results:
            label_idx = result[0]
            bbox = result[1][:4].tolist()
            score = result[1][-1]
            label_name = self.classes[label_idx]

            dict_result = {'Label': label_name, 'Bbox': bbox, 'Confidence': score}
            json_dict['Annotations'].append(dict_result)

        return json_dict
