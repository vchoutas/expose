
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met


def project_to_pixels(xyz, kk):
    """Project a single point in space into the image"""
    xx, yy, zz = np.dot(kk, xyz)
    uu = int(xx / zz)
    vv = int(yy / zz)

    return uu, vv


def project_3d(box_obj, kk):
    """
    Project a 3D bounding box into the image plane using the central corners
    """
    box_2d = []
    # Obtain the 3d points of the box
    xc, yc, zc = box_obj.center
    ww, _, hh, = box_obj.wlh

    # Points corresponding to a box at the z of the center
    x1 = xc - ww/2
    y1 = yc - hh/2  # Y axis directed below
    x2 = xc + ww/2
    y2 = yc + hh/2
    xyz1 = np.array([x1, y1, zc])
    xyz2 = np.array([x2, y2, zc])
    corners_3d = np.array([xyz1, xyz2])

    # Project them and convert into pixel coordinates
    for xyz in corners_3d:
        xx, yy, zz = np.dot(kk, xyz)
        uu = xx / zz
        vv = yy / zz
        box_2d.append(uu)
        box_2d.append(vv)

    return box_2d


def get_keypoints(keypoints, mode):
    """
    Extract center, shoulder or hip points of a keypoint
    Input --> list or torch/numpy tensor [(m, 3, 17) or (3, 17)]
    Output --> torch.tensor [(m, 2)]
    """
    if isinstance(keypoints, (list, np.ndarray)):
        keypoints = torch.tensor(keypoints)
    if len(keypoints.size()) == 2:  # add batch dim
        keypoints = keypoints.unsqueeze(0)
    assert len(keypoints.size()) == 3 and keypoints.size()[1] == 3, "tensor dimensions not recognized"
    assert mode in ['center', 'bottom', 'head', 'shoulder', 'hip', 'ankle']

    kps_in = keypoints[:, 0:2, :]  # (m, 2, 17)
    if mode == 'center':
        kps_max, _ = kps_in.max(2)  # returns value, indices
        kps_min, _ = kps_in.min(2)
        kps_out = (kps_max - kps_min) / 2 + kps_min   # (m, 2) as keepdims is False

    elif mode == 'bottom':  # bottom center for kitti evaluation
        kps_max, _ = kps_in.max(2)
        kps_min, _ = kps_in.min(2)
        kps_out_x = (kps_max[:, 0:1] - kps_min[:, 0:1]) / 2 + kps_min[:, 0:1]
        kps_out_y = kps_max[:, 1:2]
        kps_out = torch.cat((kps_out_x, kps_out_y), -1)

    elif mode == 'head':
        kps_out = kps_in[:, :, 0:5].mean(2)

    elif mode == 'shoulder':
        kps_out = kps_in[:, :, 5:7].mean(2)

    elif mode == 'hip':
        kps_out = kps_in[:, :, 11:13].mean(2)

    elif mode == 'ankle':
        kps_out = kps_in[:, :, 15:17].mean(2)

    return kps_out  # (m, 2)


def transform_kp(kps, tr_mode):
    """Apply different transformations to the keypoints based on the tr_mode"""

    assert tr_mode in ("None", "singularity", "upper", "lower", "horizontal", "vertical", "lateral",
                       'shoulder', 'knee', 'upside', 'falling', 'random')

    uu_c, vv_c = get_keypoints(kps, mode='center')

    if tr_mode == "None":
        return kps

    if tr_mode == "singularity":
        uus = [uu_c for uu in kps[0]]
        vvs = [vv_c for vv in kps[1]]

    elif tr_mode == "vertical":
        uus = [uu_c for uu in kps[0]]
        vvs = kps[1]

    elif tr_mode == 'horizontal':
        uus = kps[0]
        vvs = [vv_c for vv in kps[1]]

    elif tr_mode == 'shoulder':
        uus = kps[0]
        vvs = kps[1][:7] + [kps[1][6] for vv in kps[1][7:]]

    elif tr_mode == 'knee':
        uus = kps[0]
        vvs = [kps[1][14] for vv in kps[1][:13]] + kps[1][13:]

    elif tr_mode == 'up':
        uus = kps[0]
        vvs = [kp - 300 for kp in kps[1]]

    elif tr_mode == 'falling':
        uus = [kps[0][16] - kp + kps[1][16] for kp in kps[1]]
        vvs = [kps[1][16] - kp + kps[0][16] for kp in kps[0]]

    elif tr_mode == 'random':
        uu_min = min(kps[0])
        uu_max = max(kps[0])
        vv_min = min(kps[1])
        vv_max = max(kps[1])
        np.random.seed(6)
        uus = np.random.uniform(uu_min, uu_max, len(kps[0])).tolist()
        vvs = np.random.uniform(vv_min, vv_max, len(kps[1])).tolist()

    return [uus, vvs, kps[2], []]


def xyz_from_distance(distances, xy_centers):
    """
    From distances and normalized image coordinates (z=1), extract the real world position xyz
    distances --> tensor (m,1) or (m) or float
    xy_centers --> tensor(m,3) or (3)
    """

    if isinstance(distances, float):
        distances = torch.tensor(distances).unsqueeze(0)
    if len(distances.size()) == 1:
        distances = distances.unsqueeze(1)
    if len(xy_centers.size()) == 1:
        xy_centers = xy_centers.unsqueeze(0)

    assert xy_centers.size()[-1] == 3 and distances.size()[-1] == 1, "Size of tensor not recognized"

    return xy_centers * distances / torch.sqrt(1 + xy_centers[:, 0:1].pow(2) + xy_centers[:, 1:2].pow(2))


def open_image(path_image):
    with open(path_image, 'rb') as f:
        pil_image = Image.open(f).convert('RGB')
        return pil_image
