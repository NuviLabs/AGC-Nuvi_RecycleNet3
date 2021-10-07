#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Nuvilabs - Luca Medeiros, luca.medeiros@nuvi-labs.com
"""

import argparse
import os.path as osp
import glob
from inference import Nuvi_RecycleNet

parser = argparse.ArgumentParser(
                    description='Nuvilabs RecycleNet')
parser.add_argument('--config_file', default='./model/model_config.py', type=str,
                    help='Model configuration file path.')
parser.add_argument('--checkpoint_file', default='./model/model_checkpoint.pth', type=str,
                    help='Model checkpoint file path.')
parser.add_argument('--img_path', default='./sample_img.jpg', type=str,
                    help='Path of image or images.')
parser.add_argument('--threshold', default=0.6, type=float,
                    help='Only boxes with the score larger than this will be detected.')
parser.add_argument('--use_tta', dest='use_tta', action='store_true',
                    help='Either use TTA to help detect images.')
parser.set_defaults(use_tta=False)

if __name__ == '__main__':

    arg = parser.parse_args()
    recyclernet = Nuvi_RecycleNet(arg.config_file,
                                  arg.checkpoint_file,
                                  arg.threshold,
                                  tta=arg.use_tta)