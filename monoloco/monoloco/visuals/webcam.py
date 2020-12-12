# pylint: disable=W0212
"""
Webcam demo application

Implementation adapted from https://github.com/vita-epfl/openpifpaf/blob/master/openpifpaf/webcam.py

"""

import time

import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from ..visuals import Printer
from ..network import PifPaf, MonoLoco
from ..network.process import preprocess_pifpaf, factory_for_gt, image_transform


def webcam(args):

    # add args.device
    args.device = torch.device('cpu')
    if torch.cuda.is_available():
        args.device = torch.device('cuda')

    # load models
    args.camera = True
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.model, device=args.device)

    # Start recording
    cam = cv2.VideoCapture(0)
    visualizer_monoloco = None

    while True:
        start = time.time()
        ret, frame = cam.read()
        image = cv2.resize(frame, None, fx=args.scale, fy=args.scale)
        height, width, _ = image.shape
        print('resized image size: {}'.format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image_cpu = image_transform(image.copy())
        processed_image = processed_image_cpu.contiguous().to(args.device, non_blocking=True)
        fields = pifpaf.fields(torch.unsqueeze(processed_image, 0))[0]
        _, _, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)

        if not ret:
            break
        key = cv2.waitKey(1)

        if key % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        pil_image = Image.fromarray(image)
        intrinsic_size = [xx * 1.3 for xx in pil_image.size]
        kk, dict_gt = factory_for_gt(intrinsic_size)  # better intrinsics for mac camera
        if visualizer_monoloco is None:  # it is, at the beginning
            visualizer_monoloco = VisualizerMonoloco(kk, args)(pil_image)  # create it with the first image
            visualizer_monoloco.send(None)

        boxes, keypoints = preprocess_pifpaf(pifpaf_out, (width, height))
        outputs, varss = monoloco.forward(keypoints, kk)
        dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dict_gt)
        print(dic_out)
        visualizer_monoloco.send((pil_image, dic_out))

        end = time.time()
        print("run-time: {:.2f} ms".format((end-start)*1000))

    cam.release()

    cv2.destroyAllWindows()


class VisualizerMonoloco:
    def __init__(self, kk, args, epistemic=False):
        self.kk = kk
        self.args = args
        self.z_max = args.z_max
        self.epistemic = epistemic
        self.output_types = args.output_types

    def __call__(self, first_image, fig_width=4.0, **kwargs):
        if 'figsize' not in kwargs:
            kwargs['figsize'] = (fig_width, fig_width * first_image.size[0] / first_image.size[1])

        printer = Printer(first_image, output_path="", kk=self.kk, output_types=self.output_types,
                          z_max=self.z_max, epistemic=self.epistemic)
        figures, axes = printer.factory_axes()

        for fig in figures:
            fig.show()

        while True:
            image, dict_ann = yield
            while axes and (axes[-1] and axes[-1].patches):  # for front -1==0, for bird/combined -1 == 1
                if axes[0]:
                    del axes[0].patches[0]
                    del axes[0].texts[0]
                if len(axes) == 2:
                    del axes[1].patches[0]
                    del axes[1].patches[0]  # the one became the 0
                    if len(axes[1].lines) > 2:
                        del axes[1].lines[2]
                        if axes[1].texts:  # in case of no text
                            del axes[1].texts[0]
            printer.draw(figures, axes, dict_ann, image)
            mypause(0.01)


def mypause(interval):
    manager = plt._pylab_helpers.Gcf.get_active()
    if manager is not None:
        canvas = manager.canvas
        if canvas.figure.stale:
            canvas.draw_idle()
        canvas.start_event_loop(interval)
    else:
        time.sleep(interval)
