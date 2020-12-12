import os
import sys
from collections import defaultdict

from PIL import Image

# Python does not consider the current directory to be a package
sys.path.insert(0, os.path.join('..', 'monoloco'))


def test_printer():
    """Draw a fake figure"""
    from monoloco.visuals.printer import Printer
    test_list = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]
    boxes = [xx + [0] for xx in test_list]
    kk = test_list
    dict_ann = defaultdict(lambda: [1., 2., 3.], xyz_real=test_list, xyz_pred=test_list, uv_shoulders=test_list,
                           boxes=boxes, boxes_gt=boxes)
    with open('docs/002282.png', 'rb') as f:
        pil_image = Image.open(f).convert('RGB')
    printer = Printer(image=pil_image, output_path=None, kk=kk, output_types=['combined'])
    figures, axes = printer.factory_axes()
    printer.draw(figures, axes, dict_ann, pil_image)
