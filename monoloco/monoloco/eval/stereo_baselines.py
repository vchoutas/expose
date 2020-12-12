
""""Generate stereo baselines for kitti evaluation"""

import warnings
from collections import defaultdict

import numpy as np

from ..utils import get_keypoints


def baselines_association(baselines, zzs, keypoints, keypoints_right, reid_features):
    """compute stereo depth for each of the given stereo baselines"""

    # Initialize variables
    zzs_stereo = defaultdict()
    cnt_stereo = defaultdict(int)

    features, features_r, keypoints, keypoints_r = factory_features(
        keypoints, keypoints_right, baselines, reid_features)

    # count maximum possible associations
    cnt_stereo['max'] = min(keypoints.shape[0], keypoints_r.shape[0])  # pylint: disable=E1136

    # Filter joints disparity and calculate avg disparity
    avg_disparities, disparities_x, disparities_y = mask_joint_disparity(keypoints, keypoints_r)

    # Iterate over each left pose
    for key in baselines:

        # Extract features of the baseline
        similarity = features_similarity(features[key], features_r[key], key, avg_disparities, zzs)

        # Compute the association based on features minimization and calculate depth
        zzs_stereo[key] = np.empty((keypoints.shape[0]))

        indices_stereo = []  # keep track of indices
        best = np.nanmin(similarity)
        while not np.isnan(best):
            idx, arg_best = np.unravel_index(np.nanargmin(similarity), similarity.shape)  # pylint: disable=W0632
            zz_stereo, flag = similarity_to_depth(avg_disparities[idx, arg_best])
            zz_mono = zzs[idx]
            similarity[idx, :] = np.nan
            indices_stereo.append(idx)

            # Filter stereo depth
            if flag and verify_stereo(zz_stereo, zz_mono, disparities_x[idx, arg_best], disparities_y[idx, arg_best]):
                zzs_stereo[key][idx] = zz_stereo
                cnt_stereo[key] += 1
                similarity[:, arg_best] = np.nan
            else:
                zzs_stereo[key][idx] = zz_mono

            best = np.nanmin(similarity)
        indices_mono = [idx for idx, _ in enumerate(zzs) if idx not in indices_stereo]
        for idx in indices_mono:
            zzs_stereo[key][idx] = zzs[idx]
        zzs_stereo[key] = zzs_stereo[key].tolist()

    return zzs_stereo, cnt_stereo


def factory_features(keypoints, keypoints_right, baselines, reid_features):

    features = defaultdict()
    features_r = defaultdict()

    for key in baselines:
        if key == 'reid':
            features[key] = np.array(reid_features[0])
            features_r[key] = np.array(reid_features[1])
        else:
            features[key] = np.array(keypoints)
            features_r[key] = np.array(keypoints_right)

    return features, features_r, np.array(keypoints), np.array(keypoints_right)


def features_similarity(features, features_r, key, avg_disparities, zzs):

    similarity = np.empty((features.shape[0], features_r.shape[0]))
    for idx, zz_mono in enumerate(zzs):
        feature = features[idx]

        if key == 'ml_stereo':
            expected_disparity = 0.54 * 721. / zz_mono
            sim_row = np.abs(expected_disparity - avg_disparities[idx])

        elif key == 'pose':
            # Zero-center the keypoints
            uv_center = np.array(get_keypoints(feature, mode='center').reshape(-1, 1))  # (1, 2) --> (2, 1)
            uv_centers_r = np.array(get_keypoints(features_r, mode='center').unsqueeze(-1))  # (m,2) --> (m, 2, 1)
            feature_0 = feature[:2, :] - uv_center
            feature_0 = feature_0.reshape(1, -1)  # (1, 34)
            features_r_0 = features_r[:, :2, :] - uv_centers_r
            features_r_0 = features_r_0.reshape(features_r_0.shape[0], -1)  # (m, 34)
            sim_row = np.linalg.norm(feature_0 - features_r_0, axis=1)

        else:
            sim_row = np.linalg.norm(feature - features_r, axis=1)

        similarity[idx] = sim_row
    return similarity


def similarity_to_depth(avg_disparity):

    try:
        zz_stereo = 0.54 * 721. / float(avg_disparity)
        flag = True
    except (ZeroDivisionError, ValueError):  # All nan-slices or zero division
        zz_stereo = np.nan
        flag = False

    return zz_stereo, flag


def mask_joint_disparity(keypoints, keypoints_r):
    """filter joints based on confidence and interquartile range of the distribution"""

    CONF_MIN = 0.3
    with warnings.catch_warnings() and np.errstate(invalid='ignore'):
        disparity_x_mask = np.empty((keypoints.shape[0], keypoints_r.shape[0], 17))
        disparity_y_mask = np.empty((keypoints.shape[0], keypoints_r.shape[0], 17))
        avg_disparity = np.empty((keypoints.shape[0], keypoints_r.shape[0]))

        for idx, kps in enumerate(keypoints):
            disparity_x = kps[0, :] - keypoints_r[:, 0, :]
            disparity_y = kps[1, :] - keypoints_r[:, 1, :]

            # Mask for low confidence
            mask_conf_left = kps[2, :] > CONF_MIN
            mask_conf_right = keypoints_r[:, 2, :] > CONF_MIN
            mask_conf = mask_conf_left & mask_conf_right
            disparity_x_conf = np.where(mask_conf, disparity_x, np.nan)
            disparity_y_conf = np.where(mask_conf, disparity_y, np.nan)

            # Mask outliers using iqr
            mask_outlier = interquartile_mask(disparity_x_conf)
            x_mask_row = np.where(mask_outlier, disparity_x_conf, np.nan)
            y_mask_row = np.where(mask_outlier, disparity_y_conf, np.nan)
            avg_row = np.nanmedian(x_mask_row, axis=1)  # ignore the nan

            # Append
            disparity_x_mask[idx] = x_mask_row
            disparity_y_mask[idx] = y_mask_row
            avg_disparity[idx] = avg_row

        return avg_disparity, disparity_x_mask, disparity_y_mask


def verify_stereo(zz_stereo, zz_mono, disparity_x, disparity_y):
    """Verify disparities based on coefficient of variation, maximum y difference and z difference wrt monoloco"""

    COV_MIN = 0.1
    y_max_difference = (50 / zz_mono)
    z_max_difference = 0.6 * zz_mono

    cov = float(np.nanstd(disparity_x) / np.abs(np.nanmean(disparity_x)))  # Coefficient of variation
    avg_disparity_y = np.nanmedian(disparity_y)

    if abs(zz_stereo - zz_mono) < z_max_difference and \
            avg_disparity_y < y_max_difference and \
            cov < COV_MIN\
            and 4 < zz_stereo < 40:
        return True
    # if not np.isnan(zz_stereo):
    #     return True
    return False


def interquartile_mask(distribution):
    quartile_1, quartile_3 = np.nanpercentile(distribution, [25, 75], axis=1)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (distribution < upper_bound.reshape(-1, 1)) & (distribution > lower_bound.reshape(-1, 1))
