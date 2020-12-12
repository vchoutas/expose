"""Extract joints annotations and match with nuScenes ground truths
"""

import os
import sys
import time
import json
import logging
from collections import defaultdict
import datetime

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

from ..utils import get_iou_matches, append_cluster, select_categories, project_3d
from ..network.process import preprocess_pifpaf, preprocess_monoloco


class PreprocessNuscenes:
    """
    Preprocess Nuscenes dataset
    """
    CAMERAS = ('CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    dic_jo = {'train': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                            clst=defaultdict(lambda: defaultdict(list))),
              'val': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                          clst=defaultdict(lambda: defaultdict(list))),
              'test': dict(X=[], Y=[], names=[], kps=[], boxes_3d=[], K=[],
                           clst=defaultdict(lambda: defaultdict(list)))
              }
    dic_names = defaultdict(lambda: defaultdict(list))

    def __init__(self, dir_ann, dir_nuscenes, dataset, iou_min):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.iou_min = iou_min
        self.dir_ann = dir_ann
        dir_out = os.path.join('data', 'arrays')
        assert os.path.exists(dir_nuscenes), "Nuscenes directory does not exists"
        assert os.path.exists(self.dir_ann), "The annotations directory does not exists"
        assert os.path.exists(dir_out), "Joints directory does not exists"

        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_joints = os.path.join(dir_out, 'joints-' + dataset + '-' + now_time + '.json')
        self.path_names = os.path.join(dir_out, 'names-' + dataset + '-' + now_time + '.json')

        self.nusc, self.scenes, self.split_train, self.split_val = factory(dataset, dir_nuscenes)

    def run(self):
        """
        Prepare arrays for training
        """
        cnt_scenes = cnt_samples = cnt_sd = cnt_ann = 0
        start = time.time()
        for ii, scene in enumerate(self.scenes):
            end_scene = time.time()
            current_token = scene['first_sample_token']
            cnt_scenes += 1
            time_left = str((end_scene - start_scene) / 60 * (len(self.scenes) - ii))[:4] if ii != 0 else "NaN"

            sys.stdout.write('\r' + 'Elaborating scene {}, remaining time {} minutes'
                             .format(cnt_scenes, time_left) + '\t\n')
            start_scene = time.time()
            if scene['name'] in self.split_train:
                phase = 'train'
            elif scene['name'] in self.split_val:
                phase = 'val'
            else:
                print("phase name not in training or validation split")
                continue

            while not current_token == "":
                sample_dic = self.nusc.get('sample', current_token)
                cnt_samples += 1

                # Extract all the sample_data tokens for each sample
                for cam in self.CAMERAS:
                    sd_token = sample_dic['data'][cam]
                    cnt_sd += 1

                    # Extract all the annotations of the person
                    name, boxes_gt, boxes_3d, dds, kk = self.extract_from_token(sd_token)

                    # Run IoU with pifpaf detections and save
                    path_pif = os.path.join(self.dir_ann, name + '.pifpaf.json')
                    exists = os.path.isfile(path_pif)

                    if exists:
                        with open(path_pif, 'r') as file:
                            annotations = json.load(file)
                            boxes, keypoints = preprocess_pifpaf(annotations, im_size=(1600, 900))
                    else:
                        continue

                    if keypoints:
                        inputs = preprocess_monoloco(keypoints, kk).tolist()

                        matches = get_iou_matches(boxes, boxes_gt, self.iou_min)
                        for (idx, idx_gt) in matches:
                            self.dic_jo[phase]['kps'].append(keypoints[idx])
                            self.dic_jo[phase]['X'].append(inputs[idx])
                            self.dic_jo[phase]['Y'].append([dds[idx_gt]])  # Trick to make it (nn,1)
                            self.dic_jo[phase]['names'].append(name)  # One image name for each annotation
                            self.dic_jo[phase]['boxes_3d'].append(boxes_3d[idx_gt])
                            self.dic_jo[phase]['K'].append(kk)
                            append_cluster(self.dic_jo, phase, inputs[idx], dds[idx_gt], keypoints[idx])
                            cnt_ann += 1
                            sys.stdout.write('\r' + 'Saved annotations {}'.format(cnt_ann) + '\t')

                current_token = sample_dic['next']

        with open(os.path.join(self.path_joints), 'w') as f:
            json.dump(self.dic_jo, f)
        with open(os.path.join(self.path_names), 'w') as f:
            json.dump(self.dic_names, f)
        end = time.time()

        print("\nSaved {} annotations for {} samples in {} scenes. Total time: {:.1f} minutes"
              .format(cnt_ann, cnt_samples, cnt_scenes, (end-start)/60))
        print("\nOutput files:\n{}\n{}\n".format(self.path_names, self.path_joints))

    def extract_from_token(self, sd_token):

        boxes_gt = []
        dds = []
        boxes_3d = []
        path_im, boxes_obj, kk = self.nusc.get_sample_data(sd_token, box_vis_level=1)  # At least one corner
        kk = kk.tolist()
        name = os.path.basename(path_im)
        for box_obj in boxes_obj:
            if box_obj.name[:6] != 'animal':
                general_name = box_obj.name.split('.')[0] + '.' + box_obj.name.split('.')[1]
            else:
                general_name = 'animal'
            if general_name in select_categories('all'):
                box = project_3d(box_obj, kk)
                dd = np.linalg.norm(box_obj.center)
                boxes_gt.append(box)
                dds.append(dd)
                box_3d = box_obj.center.tolist() + box_obj.wlh.tolist()
                boxes_3d.append(box_3d)
                self.dic_names[name]['boxes'].append(box)
                self.dic_names[name]['dds'].append(dd)
                self.dic_names[name]['K'] = kk

        return name, boxes_gt, boxes_3d, dds, kk


def factory(dataset, dir_nuscenes):
    """Define dataset type and split training and validation"""

    assert dataset in ['nuscenes', 'nuscenes_mini', 'nuscenes_teaser']
    if dataset == 'nuscenes_mini':
        version = 'v1.0-mini'
    else:
        version = 'v1.0-trainval'

    nusc = NuScenes(version=version, dataroot=dir_nuscenes, verbose=True)
    scenes = nusc.scene

    if dataset == 'nuscenes_teaser':
        with open("splits/nuscenes_teaser_scenes.txt", "r") as file:
            teaser_scenes = file.read().splitlines()
        scenes = [scene for scene in scenes if scene['token'] in teaser_scenes]
        with open("splits/split_nuscenes_teaser.json", "r") as file:
            dic_split = json.load(file)
        split_train = [scene['name'] for scene in scenes if scene['token'] in dic_split['train']]
        split_val = [scene['name'] for scene in scenes if scene['token'] in dic_split['val']]
    else:
        split_scenes = splits.create_splits_scenes()
        split_train, split_val = split_scenes['train'], split_scenes['val']

    return nusc, scenes, split_train, split_val
