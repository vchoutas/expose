
import math

import numpy as np


def get_calibration(path_txt):
    """Read calibration parameters from txt file:
    For the left color camera we use P2 which is K * [I|t]

    P = [fu, 0, x0, fu*t1-x0*t3
         0, fv, y0, fv*t2-y0*t3
         0, 0,  1,          t3]

    check also http://ksimek.github.io/2013/08/13/intrinsic/

    Simple case test:
    xyz = np.array([2, 3, 30, 1]).reshape(4, 1)
    xyz_2 = xyz[0:-1] + tt
    uv_temp = np.dot(kk, xyz_2)
    uv_1 = uv_temp / uv_temp[-1]
    kk_1 = np.linalg.inv(kk)
    xyz_temp2 = np.dot(kk_1, uv_1)
    xyz_new_2 = xyz_temp2 * xyz_2[2]
    xyz_fin_2 = xyz_new_2 - tt
    """

    with open(path_txt, "r") as ff:
        file = ff.readlines()
    p2_str = file[2].split()[1:]
    p2_list = [float(xx) for xx in p2_str]
    p2 = np.array(p2_list).reshape(3, 4)

    p3_str = file[3].split()[1:]
    p3_list = [float(xx) for xx in p3_str]
    p3 = np.array(p3_list).reshape(3, 4)

    kk, tt = get_translation(p2)
    kk_right, tt_right = get_translation(p3)

    return [kk, tt], [kk_right, tt_right]


def get_translation(pp):
    """Separate intrinsic matrix from translation and convert in lists"""

    kk = pp[:, :-1]
    f_x = kk[0, 0]
    f_y = kk[1, 1]
    x0, y0 = kk[2, 0:2]
    aa, bb, t3 = pp[0:3, 3]
    t1 = float((aa - x0*t3) / f_x)
    t2 = float((bb - y0*t3) / f_y)
    tt = [t1, t2, float(t3)]
    return kk.tolist(), tt


def get_simplified_calibration(path_txt):

    with open(path_txt, "r") as ff:
        file = ff.readlines()

    for line in file:
        if line[:4] == 'K_02':
            kk_str = line[4:].split()[1:]
            kk_list = [float(xx) for xx in kk_str]
            kk = np.array(kk_list).reshape(3, 3).tolist()
            return kk

    raise ValueError('Matrix K_02 not found in the file')


def check_conditions(line, category, method, thresh=0.3):
    """Check conditions of our or m3d txt file"""

    check = False
    assert category in ['pedestrian', 'cyclist', 'all']

    if method == 'gt':
        if category == 'all':
            categories_gt = ['Pedestrian', 'Person_sitting', 'Cyclist']
        else:
            categories_gt = [category.upper()[0] + category[1:]]  # Upper case names
        if line.split()[0] in categories_gt:
            check = True

    elif method in ('m3d', '3dop'):
        conf = float(line[15])
        if line[0] == category and conf >= thresh:
            check = True

    elif method == 'monodepth':
        check = True

    else:
        zz = float(line[13])
        conf = float(line[15])
        if conf >= thresh and 0.5 < zz < 70:
            check = True

    return check


def get_category(box, trunc, occ):

    hh = box[3] - box[1]
    if hh >= 40 and trunc <= 0.15 and occ <= 0:
        cat = 'easy'
    elif trunc <= 0.3 and occ <= 1 and hh >= 25:
        cat = 'moderate'
    elif trunc <= 0.5 and occ <= 2 and hh >= 25:
        cat = 'hard'
    else:
        cat = 'excluded'
    return cat


def split_training(names_gt, path_train, path_val):
    """Split training and validation images"""
    set_gt = set(names_gt)
    set_train = set()
    set_val = set()

    with open(path_train, "r") as f_train:
        for line in f_train:
            set_train.add(line[:-1] + '.txt')
    with open(path_val, "r") as f_val:
        for line in f_val:
            set_val.add(line[:-1] + '.txt')

    set_train = tuple(set_gt.intersection(set_train))
    set_val = tuple(set_gt.intersection(set_val))
    assert set_train and set_val, "No validation or training annotations"
    return set_train, set_val


def parse_ground_truth(path_gt, category):
    """Parse KITTI ground truth files"""
    boxes_gt = []
    dds_gt = []
    zzs_gt = []
    truncs_gt = []  # Float from 0 to 1
    occs_gt = []  # Either 0,1,2,3 fully visible, partly occluded, largely occluded, unknown
    boxes_3d = []

    with open(path_gt, "r") as f_gt:
        for line_gt in f_gt:
            if check_conditions(line_gt, category, method='gt'):
                truncs_gt.append(float(line_gt.split()[1]))
                occs_gt.append(int(line_gt.split()[2]))
                boxes_gt.append([float(x) for x in line_gt.split()[4:8]])
                loc_gt = [float(x) for x in line_gt.split()[11:14]]
                wlh = [float(x) for x in line_gt.split()[8:11]]
                boxes_3d.append(loc_gt + wlh)
                zzs_gt.append(loc_gt[2])
                dds_gt.append(math.sqrt(loc_gt[0] ** 2 + loc_gt[1] ** 2 + loc_gt[2] ** 2))

    return boxes_gt, boxes_3d, dds_gt, zzs_gt, truncs_gt, occs_gt
