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



