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


import json
import numpy as np


def read_keypoints(keypoint_fn, use_hands=True, use_face=True,
                   use_face_contour=True):
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    all_keypoints = []
    for idx, person_data in enumerate(data['people']):
        body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                  dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])

        left_hand_keyps = person_data.get('hand_left_keypoints_2d', [])
        if len(left_hand_keyps) < 1:
            left_hand_keyps = [0] * (21 * 3)
        left_hand_keyps = np.array(
            left_hand_keyps, dtype=np.float32).reshape([-1, 3])

        right_hand_keyps = person_data.get('hand_right_keypoints_2d', [])
        if len(right_hand_keyps) < 1:
            right_hand_keyps = [0] * (21 * 3)
        right_hand_keyps = np.array(
            right_hand_keyps, dtype=np.float32).reshape([-1, 3])

        face_keypoints = person_data.get('face_keypoints_2d', [])
        if len(face_keypoints) < 1:
            face_keypoints = [0] * (70 * 3)

        face_keypoints = np.array(
            face_keypoints,
            dtype=np.float32).reshape([-1, 3])

        face_keypoints = face_keypoints[:-2]

        all_keypoints.append(
            np.concatenate([
                body_keypoints,
                left_hand_keyps, right_hand_keyps,
                face_keypoints], axis=0)
        )

    if len(all_keypoints) < 1:
        return None
    all_keypoints = np.stack(all_keypoints)
    return all_keypoints
