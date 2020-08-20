import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_head import CameraHead
from .camera_projection import build_cam_proj


def build_camera_head(cfg, feat_dim):
    return CameraHead(cfg, feat_dim)
