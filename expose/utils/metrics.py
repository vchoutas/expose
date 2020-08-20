# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Dict
import numpy as np

import torch
from loguru import logger

from .np_utils import np2o3d_pcl


class NoAligment(object):
    def __init__(self):
        super(NoAligment, self).__init__()

    def __repr__(self):
        return 'NoAlignment'

    def __call__(self, S1, S2):
        return S1


class ProcrustesAlignment(object):
    def __init__(self):
        super(ProcrustesAlignment, self).__init__()

    def __repr__(self):
        return 'ProcrustesAlignment'

    def __call__(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrustes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat


class ProcrustesAlignmentMPJPE(ProcrustesAlignment):
    def __init__(self, fscore_thresholds=None):
        super(ProcrustesAlignmentMPJPE, self).__init__()
        self.fscore_thresholds = fscore_thresholds

    def __repr__(self):
        msg = [super(ProcrustesAlignment).__repr__()]
        if self.fscore_thresholds is not None:
            msg.append(
                'F-Score thresholds: ' +
                f'(mm), '.join(map(lambda x: f'{x * 1000}',
                                   self.fscore_thresholds))
            )
        return '\n'.join(msg)

    def __call__(self, est_points, gt_points):
        aligned_est_points = super(ProcrustesAlignmentMPJPE, self).__call__(
            est_points, gt_points)

        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_points, gt_points, thresh)
        return {
            'point': mpjpe(aligned_est_points, gt_points),
            'fscore': fscore
        }


class ScaleAlignment(object):
    def __init__(self):
        super(ScaleAlignment, self).__init__()

    def __repr__(self):
        return 'ScaleAlignment'

    def __call__(self, S1, S2):
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert(S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1**2)
        var2 = np.sum(X2**2)

        # 5. Recover scale.
        scale = np.sqrt(var2 / var1)

        # 6. Recover translation.
        t = mu2 - scale * (mu1)

        # 7. Error:
        S1_hat = scale * S1 + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat


class RootAlignmentMPJPE(object):
    def __init__(self, root=0, fscore_thresholds=None):
        super(RootAlignmentMPJPE, self).__init__()
        self.root = root
        self.fscore_thresholds = fscore_thresholds

    def align_by_root(self, joints):
        root_joint = joints[self.root, :]
        return {'joints': joints - root_joint, 'root': root_joint}

    def __call__(self, gt, est):
        gt_out = self.align_by_root(gt)
        est_out = self.align_by_root(est)

        aligned_gt_joints = gt_out['joints']
        aligned_est_joints = est_out['joints']
        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_joints, aligned_gt_joints, thresh)

        return {
            'point': mpjpe(aligned_est_joints, aligned_gt_joints),
            'fscore': fscore
        }


class PelvisAlignment(object):
    def __init__(self, hips_idxs=None):
        super(PelvisAlignment, self).__init__()
        if hips_idxs is None:
            hips_idxs = [2, 3]
        self.hips_idxs = hips_idxs

    def align_by_pelvis(self, joints):
        pelvis = joints[self.hips_idxs, :].mean(axis=0, keepdims=True)
        return {'joints': joints - pelvis, 'pelvis': pelvis}

    def __call__(self, gt, est):
        gt_out = self.align_by_pelvis(gt)
        est_out = self.align_by_pelvis(est)

        aligned_gt_joints = gt_out['joints']
        aligned_est_joints = est_out['joints']

        return aligned_gt_joints, aligned_est_joints


class PelvisAlignmentMPJPE(PelvisAlignment):
    def __init__(self, fscore_thresholds=None):
        super(PelvisAlignmentMPJPE, self).__init__()
        self.fscore_thresholds = fscore_thresholds

    def __repr__(self):
        msg = [super(PelvisAlignmentMPJPE).__repr__()]
        if self.fscore_thresholds is not None:
            msg.append(
                'F-Score thresholds: ' +
                f'(mm), '.join(map(lambda x: f'{x * 1000}',
                                   self.fscore_thresholds))
            )
        return '\n'.join(msg)

    def __call__(self, est_points, gt_points):
        aligned_gt_points, aligned_est_points = super(
            PelvisAlignmentMPJPE, self).__call__(gt_points, est_points)

        fscore = {}
        if self.fscore_thresholds is not None:
            for thresh in self.fscore_thresholds:
                fscore[thresh] = point_fscore(
                    aligned_est_points, gt_points, thresh)
        return {
            'point': mpjpe(aligned_est_points, aligned_gt_points),
            'fscore': fscore
        }


def mpjpe(input_joints, target_joints):
    ''' Calculate mean per-joint point error

    Parameters
    ----------
        input_joints: numpy.array, Jx3
            The joints predicted by the model
        target_joints: numpy.array, Jx3
            The ground truth joints
    Returns
    -------
        numpy.array, BxJ
            The per joint point error for each element in the batch
    '''

    return np.sqrt(np.power(input_joints - target_joints, 2).sum(axis=-1))


def vertex_to_vertex_error(input_vertices, target_vertices):
    return np.sqrt(np.power(input_vertices - target_vertices, 2).sum(axis=-1))


def point_fscore(
        pred: torch.Tensor,
        gt: torch.Tensor,
        thresh: float) -> Dict[str, float]:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()

    pred_pcl = np2o3d_pcl(pred)
    gt_pcl = np2o3d_pcl(gt)

    gt_to_pred = np.asarray(gt_pcl.compute_point_cloud_distance(pred_pcl))
    pred_to_gt = np.asarray(pred_pcl.compute_point_cloud_distance(gt_pcl))

    recall = (pred_to_gt < thresh).sum() / len(pred_to_gt)
    precision = (gt_to_pred < thresh).sum() / len(gt_to_pred)
    if recall + precision > 0.0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0.0

    return {
        'fscore': fscore,
        'precision': precision,
        'recall': recall,
    }
