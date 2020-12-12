
import os
import json

import torch
from PIL import Image
from openpifpaf import show

from .visuals.printer import Printer
from .network import PifPaf, ImageList, MonoLoco
from .network.process import factory_for_gt, preprocess_pifpaf


def predict(args):

    cnt = 0

    # load pifpaf and monoloco models
    pifpaf = PifPaf(args)
    monoloco = MonoLoco(model=args.model, device=args.device, n_dropout=args.n_dropout, p_dropout=args.dropout)

    # data
    data = ImageList(args.images, scale=args.scale)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=1, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers)

    for idx, (image_paths, image_tensors, processed_images_cpu) in enumerate(data_loader):
        images = image_tensors.permute(0, 2, 3, 1)

        processed_images = processed_images_cpu.to(args.device, non_blocking=True)
        fields_batch = pifpaf.fields(processed_images)

        # unbatch
        for image_path, image, processed_image_cpu, fields in zip(
                image_paths, images, processed_images_cpu, fields_batch):

            if args.output_directory is None:
                output_path = image_path
            else:
                file_name = os.path.basename(image_path)
                output_path = os.path.join(args.output_directory, file_name)
            print('image', idx, image_path, output_path)

            keypoint_sets, scores, pifpaf_out = pifpaf.forward(image, processed_image_cpu, fields)
            pifpaf_outputs = [keypoint_sets, scores, pifpaf_out]  # keypoints_sets and scores for pifpaf printing
            images_outputs = [image]  # List of 1 or 2 elements with pifpaf tensor (resized) and monoloco original image

            if 'monoloco' in args.networks:
                im_size = (float(image.size()[1] / args.scale),
                           float(image.size()[0] / args.scale))  # Width, Height (original)

                # Extract calibration matrix and ground truth file if present
                with open(image_path, 'rb') as f:
                    pil_image = Image.open(f).convert('RGB')
                    images_outputs.append(pil_image)

                im_name = os.path.basename(image_path)

                kk, dic_gt = factory_for_gt(im_size, name=im_name, path_gt=args.path_gt)

                # Preprocess pifpaf outputs and run monoloco
                boxes, keypoints = preprocess_pifpaf(pifpaf_out, im_size)
                outputs, varss = monoloco.forward(keypoints, kk)
                dic_out = monoloco.post_process(outputs, varss, boxes, keypoints, kk, dic_gt)

            else:
                dic_out = None
                kk = None

            factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=dic_out, kk=kk)
            print('Image {}\n'.format(cnt) + '-' * 120)
            cnt += 1


def factory_outputs(args, images_outputs, output_path, pifpaf_outputs, dic_out=None, kk=None):
    """Output json files or images according to the choice"""

    # Save json file
    if 'pifpaf' in args.networks:
        keypoint_sets, scores, pifpaf_out = pifpaf_outputs[:]

        # Visualizer
        keypoint_painter = show.KeypointPainter(show_box=False)
        skeleton_painter = show.KeypointPainter(show_box=False, color_connections=True,
                                                markersize=1, linewidth=4)

        if 'json' in args.output_types and keypoint_sets.size > 0:
            with open(output_path + '.pifpaf.json', 'w') as f:
                json.dump(pifpaf_out, f)

        if 'keypoints' in args.output_types:
            with show.image_canvas(images_outputs[0],
                                   output_path + '.keypoints.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                keypoint_painter.keypoints(ax, keypoint_sets)

        if 'skeleton' in args.output_types:
            with show.image_canvas(images_outputs[0],
                                   output_path + '.skeleton.png',
                                   show=args.show,
                                   fig_width=args.figure_width,
                                   dpi_factor=args.dpi_factor) as ax:
                skeleton_painter.keypoints(ax, keypoint_sets, scores=scores)

    if 'monoloco' in args.networks:
        if any((xx in args.output_types for xx in ['front', 'bird', 'combined'])):
            epistemic = False
            if args.n_dropout > 0:
                epistemic = True

            if dic_out['boxes']:  # Only print in case of detections
                printer = Printer(images_outputs[1], output_path, kk, output_types=args.output_types
                                  , z_max=args.z_max, epistemic=epistemic)
                figures, axes = printer.factory_axes()
                printer.draw(figures, axes, dic_out, images_outputs[1], draw_box=args.draw_box,
                             save=True, show=args.show)

        if 'json' in args.output_types:
            with open(os.path.join(output_path + '.monoloco.json'), 'w') as ff:
                json.dump(dic_out, ff)
