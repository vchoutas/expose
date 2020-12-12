# pylint: disable=R0915

import math
import itertools
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ..utils import get_task_error, get_pixel_error


def show_results(dic_stats, show=False, save=False, stereo=False):
    """
    Visualize error as function of the distance and compare it with target errors based on human height analyses
    """

    dir_out = 'docs'
    phase = 'test'
    x_min = 0
    x_max = 38
    y_min = 0
    y_max = 4.7
    xx = np.linspace(0, 60, 100)
    excl_clusters = ['all', '50', '>50', 'easy', 'moderate', 'hard']
    clusters = tuple([clst for clst in dic_stats[phase]['monoloco'] if clst not in excl_clusters])
    yy_gender = get_task_error(xx)

    styles = printing_styles(stereo)
    for idx_style, (key, style) in enumerate(styles.items()):
        plt.figure(idx_style)
        plt.grid(linewidth=0.2)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel("Ground-truth distance [m]")
        plt.ylabel("Average localization error [m]")
        for idx, method in enumerate(style['methods']):
            errs = [dic_stats[phase][method][clst]['mean'] for clst in clusters]
            assert errs, "method %s empty" % method
            xxs = get_distances(clusters)

            plt.plot(xxs, errs, marker=style['mks'][idx], markersize=style['mksizes'][idx], linewidth=style['lws'][idx],
                     label=style['labels'][idx], linestyle=style['lstyles'][idx], color=style['colors'][idx])
        plt.plot(xx, yy_gender, '--', label="Task error", color='lightgreen', linewidth=2.5)
        if key == 'stereo':
            yy_stereo = get_pixel_error(xx)
            plt.plot(xx, yy_stereo, linewidth=1.7, color='k', label='Pixel error')

        plt.legend(loc='upper left')
        if save:
            path_fig = os.path.join(dir_out, 'results_' + key + '.png')
            plt.savefig(path_fig)
            print("Figure of results " + key + " saved in {}".format(path_fig))
        if show:
            plt.show()
        plt.close()


def show_spread(dic_stats, show=False, save=False):
    """Predicted confidence intervals and task error as a function of ground-truth distance"""

    phase = 'test'
    dir_out = 'docs'
    excl_clusters = ['all', '50', '>50', 'easy', 'moderate', 'hard']
    clusters = tuple([clst for clst in dic_stats[phase]['our'] if clst not in excl_clusters])

    plt.figure(2)
    fig, ax = plt.subplots(2, sharex=True)
    plt.xlabel("Distance [m]")
    plt.ylabel("Aleatoric uncertainty [m]")
    ar = 0.5  # Change aspect ratio of ellipses
    scale = 1.5  # Factor to scale ellipses
    rec_c = 0  # Center of the rectangle
    plots_line = True

    bbs = np.array([dic_stats[phase]['our'][key]['std_ale'] for key in clusters])
    xxs = get_distances(clusters)
    yys = get_task_error(np.array(xxs))
    ax[1].plot(xxs, bbs, marker='s', color='b', label="Spread b")
    ax[1].plot(xxs, yys, '--', color='lightgreen', label="Task error", linewidth=2.5)
    yys_up = [rec_c + ar / 2 * scale * yy for yy in yys]
    bbs_up = [rec_c + ar / 2 * scale * bb for bb in bbs]
    yys_down = [rec_c - ar / 2 * scale * yy for yy in yys]
    bbs_down = [rec_c - ar / 2 * scale * bb for bb in bbs]

    if plots_line:
        ax[0].plot(xxs, yys_up, '--', color='lightgreen', markersize=5, linewidth=1.4)
        ax[0].plot(xxs, yys_down, '--', color='lightgreen', markersize=5, linewidth=1.4)
        ax[0].plot(xxs, bbs_up, marker='s', color='b', markersize=5, linewidth=0.7)
        ax[0].plot(xxs, bbs_down, marker='s', color='b', markersize=5, linewidth=0.7)

    for idx, xx in enumerate(xxs):
        te = Ellipse((xx, rec_c), width=yys[idx] * ar * scale, height=scale, angle=90, color='lightgreen', fill=True)
        bi = Ellipse((xx, rec_c), width=bbs[idx] * ar * scale, height=scale, angle=90, color='b', linewidth=1.8,
                     fill=False)

        ax[0].add_patch(te)
        ax[0].add_patch(bi)

    fig.subplots_adjust(hspace=0.1)
    plt.setp([aa.get_yticklabels() for aa in fig.axes[:-1]], visible=False)
    plt.legend()
    if save:
        path_fig = os.path.join(dir_out, 'spread.png')
        plt.savefig(path_fig)
        print("Figure of confidence intervals saved in {}".format(path_fig))
    if show:
        plt.show()
    plt.close()


def show_task_error(show, save):
    """Task error figure"""
    plt.figure(3)
    dir_out = 'docs'
    xx = np.linspace(0.1, 50, 100)
    mu_men = 178
    mu_women = 165
    mu_child_m = 164
    mu_child_w = 156
    mm_gmm, mm_male, mm_female = calculate_gmm()
    mm_young_male = mm_male + (mu_men - mu_child_m) / mu_men
    mm_young_female = mm_female + (mu_women - mu_child_w) / mu_women
    yy_male = target_error(xx, mm_male)
    yy_female = target_error(xx, mm_female)
    yy_young_male = target_error(xx, mm_young_male)
    yy_young_female = target_error(xx, mm_young_female)
    yy_gender = target_error(xx, mm_gmm)
    yy_stereo = get_pixel_error(xx)
    plt.grid(linewidth=0.3)
    plt.plot(xx, yy_young_male, linestyle='dotted', linewidth=2.1, color='b', label='Adult/young male')
    plt.plot(xx, yy_young_female, linestyle='dotted', linewidth=2.1, color='darkorange', label='Adult/young female')
    plt.plot(xx, yy_gender, '--', color='lightgreen', linewidth=2.8, label='Generic adult (task error)')
    plt.plot(xx, yy_female, '-.', linewidth=1.7, color='darkorange', label='Adult female')
    plt.plot(xx, yy_male, '-.', linewidth=1.7, color='b', label='Adult male')
    plt.plot(xx, yy_stereo, linewidth=1.7, color='k', label='Pixel error')
    plt.xlim(np.min(xx), np.max(xx))
    plt.xlabel("Ground-truth distance from the camera $d_{gt}$ [m]")
    plt.ylabel("Localization error $\hat{e}$  due to human height variation [m]")  # pylint: disable=W1401
    plt.legend(loc=(0.01, 0.55))  # Location from 0 to 1 from lower left
    if save:
        path_fig = os.path.join(dir_out, 'task_error.png')
        plt.savefig(path_fig)
        print("Figure of task error saved in {}".format(path_fig))
    if show:
        plt.show()
    plt.close()


def show_method(save):
    """ method figure"""
    dir_out = 'docs'
    std_1 = 0.75
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ell_3 = Ellipse((0, 2), width=std_1 * 2, height=0.3, angle=-90, color='b', fill=False, linewidth=2.5)
    ell_4 = Ellipse((0, 2), width=std_1 * 3, height=0.3, angle=-90, color='r', fill=False,
                    linestyle='dashed', linewidth=2.5)
    ax.add_patch(ell_4)
    ax.add_patch(ell_3)
    plt.plot(0, 2, marker='o', color='skyblue', markersize=9)
    plt.plot([0, 3], [0, 4], 'k--')
    plt.plot([0, -3], [0, 4], 'k--')
    plt.xlim(-3, 3)
    plt.ylim(0, 3.5)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('X [m]')
    plt.ylabel('Z [m]')
    if save:
        path_fig = os.path.join(dir_out, 'output_method.png')
        plt.savefig(path_fig)
        print("Figure of method saved in {}".format(path_fig))


def target_error(xx, mm):
    return mm * xx


def calculate_gmm():
    dist_gmm, dist_male, dist_female = height_distributions()
    # get_percentile(dist_gmm)
    mu_gmm = np.mean(dist_gmm)
    mm_gmm = np.mean(np.abs(1 - mu_gmm / dist_gmm))
    mm_male = np.mean(np.abs(1 - np.mean(dist_male) / dist_male))
    mm_female = np.mean(np.abs(1 - np.mean(dist_female) / dist_female))

    print("Mean of GMM distribution: {:.4f}".format(mu_gmm))
    print("coefficient for gmm: {:.4f}".format(mm_gmm))
    print("coefficient for men: {:.4f}".format(mm_male))
    print("coefficient for women: {:.4f}".format(mm_female))
    return mm_gmm, mm_male, mm_female


def get_confidence(xx, zz, std):
    theta = math.atan2(zz, xx)

    delta_x = std * math.cos(theta)
    delta_z = std * math.sin(theta)
    return (xx - delta_x, xx + delta_x), (zz - delta_z, zz + delta_z)


def get_distances(clusters):
    """Extract distances as intermediate values between 2 clusters"""

    clusters_ext = list(clusters)
    clusters_ext.insert(0, str(0))
    distances = []
    for idx, _ in enumerate(clusters_ext[:-1]):
        clst_0 = float(clusters_ext[idx])
        clst_1 = float(clusters_ext[idx + 1])
        distances.append((clst_1 - clst_0) / 2 + clst_0)
    return tuple(distances)


def get_confidence_points(confidences, distances, errors):
    confidence_points = []
    distance_points = []
    for idx, dd in enumerate(distances):
        conf_perc = confidences[idx]
        confidence_points.append(errors[idx] + conf_perc)
        confidence_points.append(errors[idx] - conf_perc)
        distance_points.append(dd)
        distance_points.append(dd)

    return distance_points, confidence_points


def height_distributions():
    mu_men = 178
    std_men = 7
    mu_women = 165
    std_women = 7
    dist_men = np.random.normal(mu_men, std_men, int(1e7))
    dist_women = np.random.normal(mu_women, std_women, int(1e7))

    dist_gmm = np.concatenate((dist_men, dist_women))
    return dist_gmm, dist_men, dist_women


def expandgrid(*itrs):
    mm = 0
    combinations = list(itertools.product(*itrs))

    for h_i, h_gt in combinations:
        mm += abs(float(1 - h_i / h_gt))

    mm /= len(combinations)

    return combinations


def plot_dist(dist_gmm, dist_men, dist_women):
    try:
        import seaborn as sns  # pylint: disable=C0415
        sns.distplot(dist_men, hist=False, rug=False, label="Men")
        sns.distplot(dist_women, hist=False, rug=False, label="Women")
        sns.distplot(dist_gmm, hist=False, rug=False, label="GMM")
        plt.xlabel("X [cm]")
        plt.ylabel("Height distributions of men and women")
        plt.legend()
        plt.show()
        plt.close()
    except ImportError:
        print("Import Seaborn first")


def get_percentile(dist_gmm):
    dd_gt = 1000
    mu_gmm = np.mean(dist_gmm)
    dist_d = dd_gt * mu_gmm / dist_gmm
    perc_d, _ = np.nanpercentile(dist_d, [18.5, 81.5])  # Laplace bi => 63%
    perc_d2, _ = np.nanpercentile(dist_d, [23, 77])
    mu_d = np.mean(dist_d)
    # mm_bi = (mu_d - perc_d) / mu_d
    # mm_test = (mu_d - perc_d2) / mu_d
    # mad_d = np.mean(np.abs(dist_d - mu_d))


def printing_styles(stereo):
    style = {'mono': {"labels": ['Mono3D', 'Geometric Baseline', 'MonoDepth', 'Our MonoLoco', '3DOP (stereo)'],
                      "methods": ['m3d_merged', 'geometric_merged', 'monodepth_merged', 'monoloco_merged',
                                  '3dop_merged'],
                      "mks": ['*', '^', 'p', 's', 'o'],
                      "mksizes": [6, 6, 6, 6, 6], "lws": [1.5, 1.5, 1.5, 2.2, 1.6],
                      "colors": ['r', 'deepskyblue', 'grey', 'b', 'darkorange'],
                      "lstyles": ['solid', 'solid', 'solid', 'solid', 'dashdot']}}
    if stereo:
        style['stereo'] = {"labels": ['3DOP', 'Pose Baseline', 'ReiD Baseline', 'Our MonoLoco (monocular)',
                                      'Our Stereo Baseline'],
                           "methods": ['3dop_merged', 'pose_merged', 'reid_merged', 'monoloco_merged',
                                       'ml_stereo_merged'],
                           "mks": ['o', '^', 'p', 's', 's'],
                           "mksizes": [6, 6, 6, 4, 6], "lws": [1.5, 1.5, 1.5, 1.2, 1.5],
                           "colors": ['darkorange', 'lightblue', 'red', 'b', 'b'],
                           "lstyles": ['solid', 'solid', 'solid', 'dashed', 'solid']}

    return style
