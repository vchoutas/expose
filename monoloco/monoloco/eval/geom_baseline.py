
import json
import logging
import math
from collections import defaultdict

import numpy as np

from ..utils import pixel_to_camera, get_keypoints

AVERAGE_Y = 0.48
CLUSTERS = ['10', '20', '30', 'all']


def geometric_baseline(joints):
    """
    List of json files --> 2 lists with mean and std for each segment and the total count of instances

    For each annotation:
    1. From gt boxes calculate the height (deltaY) for the segments head, shoulder, hip, ankle
    2. From mask boxes calculate distance of people using average height of people and real pixel height

    For left-right ambiguities we chose always the average of the joints

    The joints are mapped from 0 to 16 in the following order:
    ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle']

    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    cnt_tot = 0
    dic_dist = defaultdict(lambda: defaultdict(list))

    # Access the joints file
    with open(joints, 'r') as ff:
        dic_joints = json.load(ff)

    # Calculate distances for all the instances in the joints dictionary
    for phase in ['train', 'val']:
        cnt = update_distances(dic_joints[phase], dic_dist, phase, AVERAGE_Y)
        cnt_tot += cnt

    # Calculate mean and std of each segment
    dic_h_means = calculate_heights(dic_dist['heights'], mode='mean')
    dic_h_stds = calculate_heights(dic_dist['heights'], mode='std')
    errors = calculate_error(dic_dist['error'])

    # Show results
    logger.info("Computed distance of {} annotations".format(cnt_tot))
    for key in dic_h_means:
        logger.info("Average height of segment {} is {:.2f} with a std of {:.2f}".
                    format(key, dic_h_means[key], dic_h_stds[key]))
    for clst in CLUSTERS:
        logger.info("Average error over the val set for clst {}: {:.2f}".format(clst, errors[clst]))
    logger.info("Joints used: {}".format(joints))


def update_distances(dic_fin, dic_dist, phase, average_y):

    # Loop over each annotation in the json file corresponding to the image
    cnt = 0
    for idx, kps in enumerate(dic_fin['kps']):

        # Extract pixel coordinates of head, shoulder, hip, ankle and and save them
        dic_uv = {mode: get_keypoints(kps, mode) for mode in ['head', 'shoulder', 'hip', 'ankle']}

        # Convert segments from pixel coordinate to camera coordinate
        kk = dic_fin['K'][idx]
        z_met = dic_fin['boxes_3d'][idx][2]

        # Create a dict with all annotations in meters
        dic_xyz = {key: pixel_to_camera(dic_uv[key], kk, z_met) for key in dic_uv}
        dic_xyz_norm = {key: pixel_to_camera(dic_uv[key], kk, 1) for key in dic_uv}

        # Compute real height
        dy_met = abs(float((dic_xyz['hip'][0][1] - dic_xyz['shoulder'][0][1])))

        # Estimate distance for a single annotation
        z_met_real = compute_distance(dic_xyz_norm['shoulder'][0], dic_xyz_norm['hip'][0], average_y,
                                      mode='real', dy_met=dy_met)
        z_met_approx = compute_distance(dic_xyz_norm['shoulder'][0], dic_xyz_norm['hip'][0], average_y, mode='average')

        # Compute distance with respect to the center of the 3D bounding box
        d_real = math.sqrt(z_met_real ** 2 + dic_fin['boxes_3d'][idx][0] ** 2 + dic_fin['boxes_3d'][idx][1] ** 2)
        d_approx = math.sqrt(z_met_approx ** 2 +
                             dic_fin['boxes_3d'][idx][0] ** 2 + dic_fin['boxes_3d'][idx][1] ** 2)

        # Update the dictionary with distance and heights metrics
        dic_dist = update_dic_dist(dic_dist, dic_xyz, d_real, d_approx, phase)
        cnt += 1

    return cnt


def compute_distance(xyz_norm_1, xyz_norm_2, average_y, mode='average', dy_met=0):
    """
    Compute distance Z of a mask annotation (solving a linear system) for 2 possible cases:
    1. knowing specific height of the annotation (head-ankle) dy_met
    2. using mean height of people (average_y)
    """
    assert mode in ('average', 'real')

    x1 = float(xyz_norm_1[0])
    y1 = float(xyz_norm_1[1])
    x2 = float(xyz_norm_2[0])
    y2 = float(xyz_norm_2[1])
    xx = (x1 + x2) / 2

    # Choose if solving for provided height or average one.
    if mode == 'average':
        cc = - average_y  # Y axis goes down
    else:
        cc = -dy_met

    # Solving the linear system Ax = b
    matrix = np.array([[y1, 0, -xx],
                       [0, -y1, 1],
                       [y2, 0, -xx],
                       [0, -y2, 1]])

    bb = np.array([cc * xx, -cc, 0, 0]).reshape(4, 1)
    xx = np.linalg.lstsq(matrix, bb, rcond=None)
    z_met = abs(np.float(xx[0][1]))  # Abs take into account specularity behind the observer

    return z_met


def update_dic_dist(dic_dist, dic_xyz, d_real, d_approx, phase):
    """ For every annotation in a single image, update the final dictionary"""

    # Update the dict with heights metric
    if phase == 'train':
        dic_dist['heights']['head'].append(float(dic_xyz['head'][0][1]))
        dic_dist['heights']['shoulder'].append(float(dic_xyz['shoulder'][0][1]))
        dic_dist['heights']['hip'].append(float(dic_xyz['hip'][0][1]))
        dic_dist['heights']['ankle'].append(float(dic_xyz['ankle'][0][1]))

    # Update the dict with distance metrics for the test phase
    if phase == 'val':
        error = abs(d_real - d_approx)

        if d_real <= 10:
            dic_dist['error']['10'].append(error)
        elif d_real <= 20:
            dic_dist['error']['20'].append(error)
        elif d_real <= 30:
            dic_dist['error']['30'].append(error)
        else:
            dic_dist['error']['>30'].append(error)

        dic_dist['error']['all'].append(error)

    return dic_dist


def calculate_heights(heights, mode):
    """
     Compute statistics of heights based on the distance
     """

    assert mode in ('mean', 'std', 'max')
    heights_fin = {}

    head_shoulder = np.array(heights['shoulder']) - np.array(heights['head'])
    shoulder_hip = np.array(heights['hip']) - np.array(heights['shoulder'])
    hip_ankle = np.array(heights['ankle']) - np.array(heights['hip'])

    if mode == 'mean':
        heights_fin['head_shoulder'] = np.float(np.mean(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.mean(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.mean(hip_ankle)) * 100

    elif mode == 'std':
        heights_fin['head_shoulder'] = np.float(np.std(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.std(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.std(hip_ankle)) * 100

    elif mode == 'max':
        heights_fin['head_shoulder'] = np.float(np.max(head_shoulder)) * 100
        heights_fin['shoulder_hip'] = np.float(np.max(shoulder_hip)) * 100
        heights_fin['hip_ankle'] = np.float(np.max(hip_ankle)) * 100

    return heights_fin


def calculate_error(dic_errors):
    """
     Compute statistics of distances based on the distance
     """
    errors = {}
    for clst in dic_errors:
        errors[clst] = np.float(np.mean(np.array(dic_errors[clst])))
    return errors
