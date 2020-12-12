
import math
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse, Circle, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..utils import pixel_to_camera, get_task_error


class Printer:
    """
    Print results on images: birds eye view and computed distance
    """
    FONTSIZE_BV = 16
    FONTSIZE = 18
    TEXTCOLOR = 'darkorange'
    COLOR_KPS = 'yellow'

    def __init__(self, image, output_path, kk, output_types, epistemic=False, z_max=30, fig_width=10):

        self.im = image
        self.kk = kk
        self.output_types = output_types
        self.epistemic = epistemic
        self.z_max = z_max  # To include ellipses in the image
        self.y_scale = 1
        self.width = self.im.size[0]
        self.height = self.im.size[1]
        self.fig_width = fig_width

        # Define the output dir
        self.output_path = output_path
        self.cmap = cm.get_cmap('jet')
        self.extensions = []

        # Define variables of the class to change for every image
        self.mpl_im0 = self.stds_ale = self.stds_epi = self.xx_gt = self.zz_gt = self.xx_pred = self.zz_pred =\
            self.dds_real = self.uv_centers = self.uv_shoulders = self.uv_kps = self.boxes = self.boxes_gt = \
            self.uv_camera = self.radius = None

    def _process_results(self, dic_ann):
        # Include the vectors inside the interval given by z_max
        self.stds_ale = dic_ann['stds_ale']
        self.stds_epi = dic_ann['stds_epi']
        self.xx_gt = [xx[0] for xx in dic_ann['xyz_real']]
        self.zz_gt = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                      for idx, xx in enumerate(dic_ann['xyz_real'])]
        self.xx_pred = [xx[0] for xx in dic_ann['xyz_pred']]
        self.zz_pred = [xx[2] if xx[2] < self.z_max - self.stds_epi[idx] else 0
                        for idx, xx in enumerate(dic_ann['xyz_pred'])]
        self.dds_real = dic_ann['dds_real']
        self.uv_shoulders = dic_ann['uv_shoulders']
        self.boxes = dic_ann['boxes']
        self.boxes_gt = dic_ann['boxes_gt']

        self.uv_camera = (int(self.im.size[0] / 2), self.im.size[1])
        self.radius = 11 / 1600 * self.width

    def factory_axes(self):
        """Create axes for figures: front bird combined"""
        axes = []
        figures = []

        #  Initialize combined figure, resizing it for aesthetic proportions
        if 'combined' in self.output_types:
            assert 'bird' and 'front' not in self.output_types, \
                "combined figure cannot be print together with front or bird ones"

            self.y_scale = self.width / (self.height * 1.8)  # Defined proportion
            if self.y_scale < 0.95 or self.y_scale > 1.05:  # allows more variation without resizing
                self.im = self.im.resize((self.width, round(self.height * self.y_scale)))
            self.width = self.im.size[0]
            self.height = self.im.size[1]
            fig_width = self.fig_width + 0.6 * self.fig_width
            fig_height = self.fig_width * self.height / self.width

            # Distinguish between KITTI images and general images
            fig_ar_1 = 1.7 if self.y_scale > 1.7 else 1.3
            width_ratio = 1.9
            self.extensions.append('.combined.png')

            fig, (ax1, ax0) = plt.subplots(1, 2, sharey=False, gridspec_kw={'width_ratios': [1, width_ratio]},
                                           figsize=(fig_width, fig_height))
            ax1.set_aspect(fig_ar_1)
            fig.set_tight_layout(True)
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0, top=1, hspace=0, wspace=0.02)

            figures.append(fig)
            assert 'front' not in self.output_types and 'bird' not in self.output_types, \
                "--combined arguments is not supported with other visualizations"

        # Initialize front figure
        elif 'front' in self.output_types:
            width = self.fig_width
            height = self.fig_width * self.height / self.width
            self.extensions.append(".front.png")
            plt.figure(0)
            fig0, ax0 = plt.subplots(1, 1, figsize=(width, height))
            fig0.set_tight_layout(True)
            figures.append(fig0)

        # Create front figure axis
        if any(xx in self.output_types for xx in ['front', 'combined']):
            ax0 = self.set_axes(ax0, axis=0)

            divider = make_axes_locatable(ax0)
            cax = divider.append_axes('right', size='3%', pad=0.05)
            bar_ticks = self.z_max // 5 + 1
            norm = matplotlib.colors.Normalize(vmin=0, vmax=self.z_max)
            scalar_mappable = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            scalar_mappable.set_array([])
            plt.colorbar(scalar_mappable, ticks=np.linspace(0, self.z_max, bar_ticks),
                         boundaries=np.arange(- 0.05, self.z_max + 0.1, .1), cax=cax, label='Z [m]')

            axes.append(ax0)
        if not axes:
            axes.append(None)

        # Initialize bird-eye-view figure
        if 'bird' in self.output_types:
            self.extensions.append(".bird.png")
            fig1, ax1 = plt.subplots(1, 1)
            fig1.set_tight_layout(True)
            figures.append(fig1)
        if any(xx in self.output_types for xx in ['bird', 'combined']):
            ax1 = self.set_axes(ax1, axis=1)  # Adding field of view
            axes.append(ax1)
        return figures, axes

    def draw(self, figures, axes, dic_out, image, draw_text=True, legend=True, draw_box=False,
             save=False, show=False):

        # Process the annotation dictionary of monoloco
        self._process_results(dic_out)

        # Draw the front figure
        num = 0
        self.mpl_im0.set_data(image)
        for idx, uv in enumerate(self.uv_shoulders):
            if any(xx in self.output_types for xx in ['front', 'combined']) and \
                 min(self.zz_pred[idx], self.zz_gt[idx]) > 0:

                color = self.cmap((self.zz_pred[idx] % self.z_max) / self.z_max)
                self.draw_circle(axes, uv, color)
                if draw_box:
                    self.draw_boxes(axes, idx, color)

                if draw_text:
                    self.draw_text_front(axes, uv, num)
                    num += 1

        # Draw the bird figure
        num = 0
        for idx, _ in enumerate(self.xx_pred):
            if any(xx in self.output_types for xx in ['bird', 'combined']) and self.zz_gt[idx] > 0:

                # Draw ground truth and predicted ellipses
                self.draw_ellipses(axes, idx)

                # Draw bird eye view text
                if draw_text:
                    self.draw_text_bird(axes, idx, num)
                    num += 1
        # Add the legend
        if legend:
            draw_legend(axes)

        # Draw, save or/and show the figures
        for idx, fig in enumerate(figures):
            fig.canvas.draw()
            if save:
                fig.savefig(self.output_path + self.extensions[idx], bbox_inches='tight')
            if show:
                fig.show()

    def draw_ellipses(self, axes, idx):
        """draw uncertainty ellipses"""
        target = get_task_error(self.dds_real[idx])
        angle_gt = get_angle(self.xx_gt[idx], self.zz_gt[idx])
        ellipse_real = Ellipse((self.xx_gt[idx], self.zz_gt[idx]), width=target * 2, height=1,
                               angle=angle_gt, color='lightgreen', fill=True, label="Task error")
        axes[1].add_patch(ellipse_real)
        if abs(self.zz_gt[idx] - self.zz_pred[idx]) > 0.001:
            axes[1].plot(self.xx_gt[idx], self.zz_gt[idx], 'kx', label="Ground truth", markersize=3)

        angle = get_angle(self.xx_pred[idx], self.zz_pred[idx])
        ellipse_ale = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_ale[idx] * 2,
                              height=1, angle=angle, color='b', fill=False, label="Aleatoric Uncertainty",
                              linewidth=1.3)
        ellipse_var = Ellipse((self.xx_pred[idx], self.zz_pred[idx]), width=self.stds_epi[idx] * 2,
                              height=1, angle=angle, color='r', fill=False, label="Uncertainty",
                              linewidth=1, linestyle='--')

        axes[1].add_patch(ellipse_ale)
        if self.epistemic:
            axes[1].add_patch(ellipse_var)

        axes[1].plot(self.xx_pred[idx], self.zz_pred[idx], 'ro', label="Predicted", markersize=3)

    def draw_boxes(self, axes, idx, color):
        ww_box = self.boxes[idx][2] - self.boxes[idx][0]
        hh_box = (self.boxes[idx][3] - self.boxes[idx][1]) * self.y_scale
        ww_box_gt = self.boxes_gt[idx][2] - self.boxes_gt[idx][0]
        hh_box_gt = (self.boxes_gt[idx][3] - self.boxes_gt[idx][1]) * self.y_scale

        rectangle = Rectangle((self.boxes[idx][0], self.boxes[idx][1] * self.y_scale),
                              width=ww_box, height=hh_box, fill=False, color=color, linewidth=3)
        rectangle_gt = Rectangle((self.boxes_gt[idx][0], self.boxes_gt[idx][1] * self.y_scale),
                                 width=ww_box_gt, height=hh_box_gt, fill=False, color='g', linewidth=2)
        axes[0].add_patch(rectangle_gt)
        axes[0].add_patch(rectangle)

    def draw_text_front(self, axes, uv, num):
        axes[0].text(uv[0] + self.radius, uv[1] * self.y_scale - self.radius, str(num),
                     fontsize=self.FONTSIZE, color=self.TEXTCOLOR, weight='bold')

    def draw_text_bird(self, axes, idx, num):
        """Plot the number in the bird eye view map"""

        std = self.stds_epi[idx] if self.stds_epi[idx] > 0 else self.stds_ale[idx]
        theta = math.atan2(self.zz_pred[idx], self.xx_pred[idx])

        delta_x = std * math.cos(theta)
        delta_z = std * math.sin(theta)

        axes[1].text(self.xx_pred[idx] + delta_x, self.zz_pred[idx] + delta_z,
                     str(num), fontsize=self.FONTSIZE_BV, color='darkorange')

    def draw_circle(self, axes, uv, color):

        circle = Circle((uv[0], uv[1] * self.y_scale), radius=self.radius, color=color, fill=True)
        axes[0].add_patch(circle)

    def set_axes(self, ax, axis):
        assert axis in (0, 1)

        if axis == 0:
            ax.set_axis_off()
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            self.mpl_im0 = ax.imshow(self.im)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        else:
            uv_max = [0., float(self.height)]
            xyz_max = pixel_to_camera(uv_max, self.kk, self.z_max)
            x_max = abs(xyz_max[0])  # shortcut to avoid oval circles in case of different kk
            ax.plot([0, x_max], [0, self.z_max], 'k--')
            ax.plot([0, -x_max], [0, self.z_max], 'k--')
            ax.set_ylim(0, self.z_max+1)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Z [m]")

        return ax


def draw_legend(axes):
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    axes[1].legend(by_label.values(), by_label.keys())


def get_angle(xx, zz):
    """Obtain the points to plot the confidence of each annotation"""

    theta = math.atan2(zz, xx)
    angle = theta * (180 / math.pi)

    return angle
