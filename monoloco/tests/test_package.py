"""Test if the main modules of the package run correctly"""

import os
import sys
import json

# Python does not consider the current directory to be a package
sys.path.insert(0, os.path.join('..', 'monoloco'))

from PIL import Image

from monoloco.train import Trainer
from monoloco.network import MonoLoco
from monoloco.network.process import preprocess_pifpaf, factory_for_gt
from monoloco.visuals.printer import Printer

JOINTS = 'tests/joints_sample.json'
PIFPAF_KEYPOINTS = 'tests/002282.png.pifpaf.json'
IMAGE = 'docs/002282.png'


def tst_trainer(joints):
    trainer = Trainer(joints=joints, epochs=150, lr=0.01)
    _ = trainer.train()
    dic_err, model = trainer.evaluate()
    return dic_err['val']['all']['mean'], model


def tst_prediction(model, path_keypoints):
    with open(path_keypoints, 'r') as f:
        pifpaf_out = json.load(f)

    kk, _ = factory_for_gt(im_size=[1240, 340])

    # Preprocess pifpaf outputs and run monoloco
    boxes, keypoints = preprocess_pifpaf(pifpaf_out)
    monoloco = MonoLoco(model)
    outputs, varss = monoloco.forward(keypoints, kk)
    dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk)
    return dic_out, kk


def tst_printer(dic_out, kk, image_path):
    """Draw a fake figure"""
    with open(image_path, 'rb') as f:
        pil_image = Image.open(f).convert('RGB')
    printer = Printer(image=pil_image, output_path='tests/test_image', kk=kk, output_types=['combined'], z_max=15)
    figures, axes = printer.factory_axes()
    printer.draw(figures, axes, dic_out, pil_image, save=True)


def test_package():

    # Training test
    val_acc, model = tst_trainer(JOINTS)
    assert val_acc < 2.5

    # Prediction test
    dic_out, kk = tst_prediction(model, PIFPAF_KEYPOINTS)
    assert dic_out['boxes'] and kk

    # Visualization test
    tst_printer(dic_out, kk, IMAGE)






