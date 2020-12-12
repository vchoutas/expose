
import random
import json
import os

import numpy as np


def get_unique_tokens(list_fin):
    """
    list of json files --> list of unique scene tokens
    """
    list_token_scene = []

    # Open one json file at a time
    for name_fin in list_fin:
        with open(name_fin, 'r') as f:
            dict_fin = json.load(f)

        # Check if the token scene is already in the list and if not add it
        if dict_fin['token_scene'] not in list_token_scene:
            list_token_scene.append(dict_fin['token_scene'])

    return list_token_scene


def split_scenes(list_token_scene, train, val, dir_main, save=False, load=True):
    """
    Split the list according tr, val percentages (test percentage is a consequence) after shuffling the order
    """

    path_split = os.path.join(dir_main, 'scenes', 'split_scenes.json')

    if save:
        random.seed(1)
        random.shuffle(list_token_scene)  # it shuffles in place
        n_scenes = len(list_token_scene)
        n_train = round(n_scenes * train / 100)
        n_val = round(n_scenes * val / 100)
        list_train = list_token_scene[0: n_train]
        list_val = list_token_scene[n_train: n_train + n_val]
        list_test = list_token_scene[n_train + n_val:]

        dic_split = {'train': list_train, 'val': list_val, 'test': list_test}
        with open(path_split, 'w') as f:
            json.dump(dic_split, f)

    if load:
        with open(path_split, 'r') as f:
            dic_split = json.load(f)

    return dic_split


def select_categories(cat):
    """
    Choose the categories to extract annotations from
    """
    assert cat in ['person', 'all', 'car', 'cyclist']

    if cat == 'person':
        categories = ['human.pedestrian']
    elif cat == 'all':
        categories = ['human.pedestrian', 'vehicle.bicycle', 'vehicle.motorcycle']
    elif cat == 'cyclist':
        categories = ['vehicle.bicycle']
    elif cat == 'car':
        categories = ['vehicle']
    return categories


def update_with_tokens(dict_gt, nusc, token_sd):
    """
    Update with tokens corresponding to the token_sd
    """

    table_sample_data = nusc.get('sample_data', token_sd)  # Extract the whole record to get the sample token
    token_sample = table_sample_data['sample_token']  # Extract the sample_token from the table
    table_sample = nusc.get('sample', token_sample)  # Get the record of the sample
    token_scene = table_sample['scene_token']
    dict_gt['token_sample_data'] = token_sd
    dict_gt['token_sample'] = token_sample
    dict_gt['token_scene'] = token_scene
    return dict_gt


def update_with_box(dict_gt, box):

    bbox = np.zeros(7, )
    flag_child = False

    # Save the 3D bbox
    bbox[0:3] = box.center
    bbox[3:6] = box.wlh
    bbox[6] = box.orientation.degrees
    dict_gt['boxes'].append(bbox.tolist()) # Save as list to be serializable by a json file
    if box.name == 'human.pedestrian.child':
        flag_child = True

    return dict_gt, flag_child
