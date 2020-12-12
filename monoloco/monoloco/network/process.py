
import json

import numpy as np
import torch
import torchvision

from ..utils import get_keypoints, pixel_to_camera
from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)


def preprocess_monoloco(keypoints, kk):

    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    return kps_out


def factory_for_gt(im_size, name=None, path_gt=None):
    """Look for ground-truth annotations file and define calibration matrix based on image size """

    try:
        with open(path_gt, 'r') as f:
            dic_names = json.load(f)
        logger.debug('-' * 120 + "\nGround-truth file opened")
    except (FileNotFoundError, TypeError):
        logger.debug('-' * 120 + "\nGround-truth file not found")
        dic_names = {}

    try:
        kk = dic_names[name]['K']
        dic_gt = dic_names[name]
        logger.debug("Matched ground-truth file!")
    except KeyError:
        dic_gt = None
        x_factor = im_size[0] / 1600
        y_factor = im_size[1] / 900
        pixel_factor = (x_factor + y_factor) / 2   # TODO remove and check it
        if im_size[0] / im_size[1] > 2.5:
            kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        else:
            kk = [[1266.4 * pixel_factor, 0., 816.27 * x_factor],
                  [0, 1266.4 * pixel_factor, 491.5 * y_factor],
                  [0., 0., 1.]]  # nuScenes calibration

        logger.debug("Using a standard calibration matrix...")

    return kk, dic_gt


def laplace_sampling(outputs, n_samples):

    # np.random.seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(outputs):
    """Unnormalize relative bi of a nunmpy array"""

    outputs[:, 1] = torch.exp(outputs[:, 1]) * outputs[:, 0]
    return outputs


def preprocess_pifpaf(annotations, im_size=None):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []

    for dic in annotations:
        box = dic['bbox']
        if box[3] < 0.5:  # Check for no detections (boxes 0,0,0,0)
            return [], []

        kps = prepare_pif_kps(dic['keypoints'])
        conf = float(np.sort(np.array(kps[2]))[-3])  # The confidence is the 3rd highest value for the keypoints

        # Add 15% for y and 20% for x
        delta_h = (box[3] - box[1]) / 7
        delta_w = (box[2] - box[0]) / 3.5
        assert delta_h > -5 and delta_w > -5, "Bounding box <=0"
        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        box.append(conf)
        boxes.append(box)
        keypoints.append(kps)

    return boxes, keypoints


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]


def image_transform(image):

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize, ])
    return transforms(image)
