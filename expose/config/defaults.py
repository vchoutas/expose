from copy import deepcopy
from loguru import logger
#  from yacs.config import CfgNode as CN
from fvcore.common.config import CfgNode as CN

from .optim_defaults import _C as optim
from .datasets_defaults import _C as datasets
from .loss_defaults import _C as losses
from .body_model import _C as body_model


def create_camera_config(node):
    node.camera = CN()
    node.camera.type = 'weak-persp'
    node.camera.pos_func = 'softplus'

    node.camera.weak_persp = CN()
    node.camera.weak_persp.regress_translation = True
    node.camera.weak_persp.regress_scale = True
    node.camera.weak_persp.regress_scale = True
    node.camera.weak_persp.mean_scale = 0.9

    node.camera.perspective = CN()
    node.camera.perspective.regress_translation = False
    node.camera.perspective.regress_rotation = False
    node.camera.perspective.regress_focal_length = False
    node.camera.perspective.focal_length = 5000
    return node.camera


def create_mlp_config(node, key='mlp'):
    if key not in node:
        node[key] = CN()

    node[key].layers = (1024, 1024)
    node[key].activ_type = 'relu'
    node[key].lrelu_slope = 0.2
    node[key].norm_type = 'none'
    node[key].num_groups = 32
    node[key].dropout = 0.0
    node[key].init_type = 'xavier'
    node[key].gain = 0.01
    node[key].bias_init = 0.0

    return node[key]


def create_conv_layers(node, key='layer'):
    if key not in node:
        node[key] = CN()

    node[key].num_layers = 5
    node[key].num_filters = 2048
    node[key].stride = 1
    return node[key]


def create_subsample_layer(node, num_layers=3, key='layer',
                           kernel_size=3, stride=2):
    if key not in node:
        node[key] = CN()

    node[key].num_filters = (512,) * num_layers
    node[key].norm_type = 'bn'
    node[key].activ_type = 'relu'
    node[key].dim = 2
    node[key].kernel_sizes = [kernel_size] * len(node[key].num_filters)
    node[key].strides = [stride] * len(node[key].num_filters)
    node[key].padding = 1
    return node[key]


def create_backbone_cfg(node, backbone_type='resnet50'):
    if 'backbone' not in node:
        node.backbone = CN()
    node.backbone.type = backbone_type
    node.backbone.pretrained = True

    node.backbone.resnet = CN()
    node.backbone.resnet.replace_stride_with_dilation = (False, False, False)

    node.backbone.fpn = CN()
    node.backbone.fpn.pooling_type = 'concat'
    node.backbone.fpn.concat = CN()
    node.backbone.fpn.concat.use_max = True
    node.backbone.fpn.concat.use_avg = True

    node.backbone.hrnet = CN()
    node.backbone.hrnet.pretrained_layers = ['*']
    node.backbone.hrnet.pretrained_path = (
        'data/'
        'network_weights/hrnet/'
        'imagenet/hrnet_w48-8ef0771d.pth'
    )

    node.backbone.hrnet.stage1 = CN()
    node.backbone.hrnet.stage1.num_modules = 1
    node.backbone.hrnet.stage1.num_branches = 1
    node.backbone.hrnet.stage1.num_blocks = [4]
    node.backbone.hrnet.stage1.num_channels = [64]
    node.backbone.hrnet.stage1.block = 'BOTTLENECK'
    node.backbone.hrnet.stage1.fuse_method = 'SUM'

    node.backbone.hrnet.stage2 = CN()
    node.backbone.hrnet.stage2.num_modules = 1
    node.backbone.hrnet.stage2.num_branches = 2
    node.backbone.hrnet.stage2.num_blocks = [4, 4]
    node.backbone.hrnet.stage2.num_channels = [48, 96]
    node.backbone.hrnet.stage2.block = 'BASIC'
    node.backbone.hrnet.stage2.fuse_method = 'SUM'

    node.backbone.hrnet.stage3 = CN()
    node.backbone.hrnet.stage3.num_modules = 4
    node.backbone.hrnet.stage3.num_branches = 3
    node.backbone.hrnet.stage3.num_blocks = [4, 4, 4]
    node.backbone.hrnet.stage3.num_channels = [48, 96, 192]
    node.backbone.hrnet.stage3.block = 'BASIC'
    node.backbone.hrnet.stage3.fuse_method = 'SUM'

    node.backbone.hrnet.stage4 = CN()
    node.backbone.hrnet.stage4.num_modules = 3
    node.backbone.hrnet.stage4.num_branches = 4
    node.backbone.hrnet.stage4.num_blocks = [4, 4, 4, 4]
    node.backbone.hrnet.stage4.num_channels = [48, 96, 192, 384]
    node.backbone.hrnet.stage4.block = 'BASIC'
    node.backbone.hrnet.stage4.fuse_method = 'SUM'

    node.backbone.hrnet.stage2.subsample = create_subsample_layer(
        node.backbone.hrnet.stage2, key='subsample', num_layers=2)
    node.backbone.hrnet.stage2.subsample.num_filters = [96, 192]
    node.backbone.hrnet.stage2.subsample.num_filters = [384]
    node.backbone.hrnet.stage2.subsample.kernel_sizes = [3]
    node.backbone.hrnet.stage2.subsample.strides = [2]

    node.backbone.hrnet.stage3.subsample = create_subsample_layer(
        node.backbone.hrnet.stage3, key='subsample', num_layers=1)
    node.backbone.hrnet.stage3.subsample.num_filters = [192, 384]
    node.backbone.hrnet.stage3.subsample.kernel_sizes = [3, 3]
    node.backbone.hrnet.stage3.subsample.strides = [2, 2]

    node.backbone.hrnet.final_conv = create_conv_layers(
        node.backbone.hrnet, key='final_conv')
    node.backbone.hrnet.final_conv.num_filters = 2048

    return node.backbone


_C = CN()

# General

_C.num_gpus = 1
_C.local_rank = 0
_C.use_cuda = True
_C.log_file = '/tmp/logs/'

_C.output_folder = 'output'
_C.summary_folder = 'summaries'
_C.results_folder = 'results'
_C.code_folder = 'code'
_C.summary_steps = 100
_C.img_summary_steps = 100
_C.hd_img_summary_steps = 1000
_C.imgs_per_row = 2
_C.save_mesh_summary = False
_C.save_part_v2v = False


_C.checkpoint_folder = 'checkpoints'
_C.checkpoint_steps = 1000
_C.eval_steps = 500
_C.fscores_thresh = CN()
_C.fscores_thresh.hand = [5.0 / 1000, 15.0 / 1000]
_C.float_dtype = 'float32'
_C.max_duration = float('inf')
_C.max_iters = float('inf')
_C.logger_level = 'info'

_C.body_vertex_ids_path = 'data/body_verts.npy'
_C.face_vertex_ids_path = 'data/SMPL-X__FLAME_vertex_ids.npy'
_C.hand_vertex_ids_path = 'data/MANO_SMPLX_vertex_ids.pkl'

_C.interactive = True
_C.is_training = True

_C.degrees = CN()
_C.degrees.body = [90, 180, 270]
_C.degrees.hand = []
_C.degrees.hand = []

_C.plot_conf_thresh = 0.3

_C.smplx_valid_verts_fn = 'data/smplx_valid_verts.npz'
_C.use_hands_for_shape = False
_C.j14_regressor_path = 'data/SMPLX_to_J14.pkl'
_C.pretrained = ''

_C.use_adv_training = False

# Network options
_C.network = CN()
_C.network.type = 'attention'
_C.network.use_sync_bn = True

# Hand noise
_C.network.hand_add_shape_noise = False
_C.network.hand_shape_std = 0.0
_C.network.hand_shape_prob = 0.0

_C.network.add_hand_pose_noise = False
_C.network.hand_pose_std = 0.0
_C.network.num_hand_components = 3
_C.network.hand_noise_prob = 0.0

_C.network.hand_randomize_global_orient = False
_C.network.hand_global_rot_max = 0.0
_C.network.hand_global_rot_min = 0.0
_C.network.hand_global_rot_noise_prob = 0.0

# Head noise
_C.network.head_add_shape_noise = False
_C.network.head_shape_std = 0.0
_C.network.head_shape_prob = 0.0

_C.network.add_expression_noise = False
_C.network.expression_std = 0.0
_C.network.expression_prob = 0.0

_C.network.add_jaw_pose_noise = False
_C.network.jaw_pose_min = 0.0
_C.network.jaw_pose_max = 0.0
_C.network.jaw_noise_prob = 0.0

_C.network.head_randomize_global_orient = False
_C.network.head_global_rot_min = 0.0
_C.network.head_global_rot_max = 0.0
_C.network.head_global_rot_noise_prob = 0.0


# Append current value to body model parameter regressors

_C.network.attention = CN()

_C.network.predict_body = True
#  _C.network.predict_hands = True
_C.network.apply_hand_network_on_body = True
_C.network.apply_hand_network_on_hands = True
#  _C.network.predict_head = True
_C.network.apply_head_network_on_body = True
_C.network.apply_head_network_on_head = True
_C.network.attention.freeze_body = False
_C.network.attention.detach_mean = False

# Get hand and head crop from the HD image
_C.network.attention.hand_from_hd = True
_C.network.attention.head_from_hd = True

_C.network.attention.condition_hand_on_body = CN()
_C.network.attention.condition_hand_on_body.wrist_pose = True
_C.network.attention.condition_hand_on_body.finger_pose = True
_C.network.attention.condition_hand_on_body.shape = True

_C.network.attention.condition_head_on_body = CN()
_C.network.attention.condition_head_on_body.neck_pose = True
_C.network.attention.condition_head_on_body.jaw_pose = True
_C.network.attention.condition_head_on_body.shape = True
_C.network.attention.condition_head_on_body.expression = True

_C.network.attention.add_hand_mean_noise = False
_C.network.attention.hand_mean_noise_mean = 0.0
_C.network.attention.hand_mean_noise_std = 0.0

_C.network.attention.add_head_mean_noise = False
_C.network.attention.head_mean_noise_mean = 0.0
_C.network.attention.head_mean_noise_std = 0.0

_C.network.attention.refine_shape_from_hands = False
_C.network.attention.refine_shape_from_head = False
_C.network.attention.hand_bbox_thresh = 0.4
_C.network.attention.head_bbox_thresh = 0.4
_C.network.attention.update_wrists = True
_C.network.attention.mask_hand_keyps = True
_C.network.attention.mask_head_keyps = True
# Merged network that regresses all parameters from a global feature vector
_C.network.attention.smplx = CN()
_C.network.attention.smplx.feature_key = 'avg_pooling'
_C.network.attention.smplx.append_params = True
_C.network.attention.smplx.num_stages = 3
_C.network.attention.smplx.pose_last_stage = True
create_camera_config(_C.network.attention.smplx)
create_mlp_config(_C.network.attention.smplx)
create_backbone_cfg(_C.network.attention.smplx)

# The configuration for the network that predicts the poses of the hand from a
# hand crop
_C.network.attention.hand = CN()
_C.network.attention.hand.scale_factor = 2.0
_C.network.attention.hand.append_params = True
_C.network.attention.hand.num_stages = 3
_C.network.attention.hand.feature_key = 'avg_pooling'
_C.network.attention.hand.pos_func = 'softplus'
_C.network.attention.hand.pose_last_stage = True
_C.network.attention.hand.use_body_shape = False
_C.network.attention.hand.vertex_idxs_path = ''
_C.network.attention.hand.return_hand_vertices_only = True

create_mlp_config(_C.network.attention.hand)
create_backbone_cfg(_C.network.attention.hand, backbone_type='resnet18')
create_camera_config(_C.network.attention.hand)
_C.network.attention.hand.camera.weak_persp.mean_scale = 10.0
# For rendering and compatibility with pre-trained models

# The configuration for the network that predicts the jaw pose and the
# expressions
_C.network.attention.head = CN()
_C.network.attention.head.scale_factor = 1.5
_C.network.attention.head.append_params = True
_C.network.attention.head.num_stages = 3
_C.network.attention.head.feature_key = 'avg_pooling'
_C.network.attention.head.pos_func = 'softplus'
_C.network.attention.head.pose_last_stage = True
_C.network.attention.head.use_body_shape = False
_C.network.attention.head.vertex_idxs_path = ''
_C.network.attention.head.return_head_vertices_only = True
# For rendering and compatibility with pre-trained models
create_mlp_config(_C.network.attention.head)
create_backbone_cfg(_C.network.attention.head, backbone_type='resnet18')
create_camera_config(_C.network.attention.head)
_C.network.attention.head.camera.weak_persp.mean_scale = 8.0

# BODY MODELS
for key in body_model:
    _C[key] = body_model[key].clone()

# Append the configuration for the optimizer
_C.optim = optim.clone()

# Add the configuration for the losses
_C.losses = losses.clone()

# Add the configuration for the datasets
_C.datasets = datasets.clone()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
