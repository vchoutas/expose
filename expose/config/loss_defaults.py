from copy import deepcopy
#  from yacs.config import CfgNode as CN
from fvcore.common.config import CfgNode as CN

_C = CN()


_C.stages_to_penalize = [-1]
_C.stages_to_regularize = [-1]

_C.body_joints_2d = CN()
_C.body_joints_2d.type = 'keypoints'
_C.body_joints_2d.robustifier = 'none'
_C.body_joints_2d.norm_type = 'l1'
_C.body_joints_2d.rho = 100.0
_C.body_joints_2d.beta = 5.0 / 100 * 2
_C.body_joints_2d.size_average = True
_C.body_joints_2d.weight = 1.0
_C.body_joints_2d.enable = 0

_C.hand_joints_2d = CN()
_C.hand_joints_2d.type = 'keypoints'
_C.hand_joints_2d.norm_type = 'l1'
_C.hand_joints_2d.robustifier = 'none'
_C.hand_joints_2d.rho = 100.0
_C.hand_joints_2d.beta = 5.0 / 100 * 2
_C.hand_joints_2d.size_average = True
_C.hand_joints_2d.weight = 1.0
_C.hand_joints_2d.enable = 0

_C.face_joints_2d = CN()
_C.face_joints_2d.type = 'keypoints'
_C.face_joints_2d.norm_type = 'l1'
_C.face_joints_2d.robustifier = 'none'
_C.face_joints_2d.rho = 100.0
_C.face_joints_2d.beta = 5.0 / 100 * 2
_C.face_joints_2d.size_average = True
_C.face_joints_2d.weight = 1.0
_C.face_joints_2d.enable = 0


_C.head_crop_keypoints = CN()
_C.head_crop_keypoints.type = 'keypoints'
_C.head_crop_keypoints.norm_type = 'l1'
_C.head_crop_keypoints.robustifier = 'none'
_C.head_crop_keypoints.rho = 100.0
_C.head_crop_keypoints.beta = 5.0 / 100 * 2
_C.head_crop_keypoints.size_average = True
_C.head_crop_keypoints.weight = 0.0
_C.head_crop_keypoints.enable = 0

_C.left_hand_crop_keypoints = CN()
_C.left_hand_crop_keypoints.type = 'keypoints'
_C.left_hand_crop_keypoints.norm_type = 'l1'
_C.left_hand_crop_keypoints.robustifier = 'none'
_C.left_hand_crop_keypoints.rho = 100.0
_C.left_hand_crop_keypoints.beta = 5.0 / 100 * 2
_C.left_hand_crop_keypoints.size_average = True
_C.left_hand_crop_keypoints.weight = 0.0
_C.left_hand_crop_keypoints.enable = 0

_C.right_hand_crop_keypoints = CN()
_C.right_hand_crop_keypoints.type = 'keypoints'
_C.right_hand_crop_keypoints.norm_type = 'l1'
_C.right_hand_crop_keypoints.robustifier = 'none'
_C.right_hand_crop_keypoints.rho = 100.0
_C.right_hand_crop_keypoints.beta = 5.0 / 100 * 2
_C.right_hand_crop_keypoints.size_average = True
_C.right_hand_crop_keypoints.weight = 0.0
_C.right_hand_crop_keypoints.enable = 0

_C.body_edge_2d = CN()
_C.body_edge_2d.norm_type = 'l2'
_C.body_edge_2d.rho = 100.0
_C.body_edge_2d.beta = 5.0 / 100 * 2
_C.body_edge_2d.size_average = True
_C.body_edge_2d.weight = 0.0
_C.body_edge_2d.enable = 0
_C.body_edge_2d.robustifier = 'none'
_C.body_edge_2d.scale = 1.0
_C.body_edge_2d.threshold = 1.0


_C.hand_edge_2d = CN()
_C.hand_edge_2d.norm_type = 'l2'
_C.hand_edge_2d.rho = 100.0
_C.hand_edge_2d.beta = 5.0 / 100 * 2
_C.hand_edge_2d.size_average = True
_C.hand_edge_2d.weight = 0.0
_C.hand_edge_2d.enable = 0
_C.hand_edge_2d.robustifier = 'none'
_C.hand_edge_2d.scale = 1.0
_C.hand_edge_2d.threshold = 1.0


_C.face_edge_2d = CN()
_C.face_edge_2d.norm_type = 'l2'
_C.face_edge_2d.rho = 100.0
_C.face_edge_2d.beta = 5.0 / 100 * 2
_C.face_edge_2d.size_average = True
_C.face_edge_2d.weight = 0.0
_C.face_edge_2d.enable = 0
_C.face_edge_2d.robustifier = 'none'
_C.face_edge_2d.scale = 1.0
_C.face_edge_2d.threshold = 1.0

_C.body_joints_3d = CN()
_C.body_joints_3d.type = 'keypoints'
_C.body_joints_3d.norm_type = 'l1'
_C.body_joints_3d.rho = 100.0
_C.body_joints_3d.beta = 5.0 / 100 * 2
_C.body_joints_3d.size_average = True
_C.body_joints_3d.weight = 0.0
_C.body_joints_3d.enable = 0


_C.hand_joints_3d = CN()
_C.hand_joints_3d.type = 'keypoints'
_C.hand_joints_3d.norm_type = 'l1'
_C.hand_joints_3d.rho = 100.0
_C.hand_joints_3d.beta = 5.0 / 100 * 2
_C.hand_joints_3d.size_average = True
_C.hand_joints_3d.weight = 0.0
_C.hand_joints_3d.enable = 500 * 1000


_C.face_joints_3d = CN()
_C.face_joints_3d.type = 'keypoints'
_C.face_joints_3d.norm_type = 'l1'
_C.face_joints_3d.rho = 100.0
_C.face_joints_3d.beta = 5.0 / 100 * 2
_C.face_joints_3d.size_average = True
_C.face_joints_3d.weight = 0.0
_C.face_joints_3d.enable = 500 * 1000


_C.shape = CN()
_C.shape.type = 'l2'
_C.shape.weight = 1.0
_C.shape.enable = 0
_C.shape.prior = CN()
_C.shape.prior.type = 'l2'
_C.shape.prior.weight = 0.0
_C.shape.prior.margin = 1.0
_C.shape.prior.norm = 'l2'
_C.shape.prior.use_vector = True
_C.shape.prior.barrier = 'log'
_C.shape.prior.epsilon = 1e-7

_C.expression = CN()
_C.expression.type = 'l2'
_C.expression.weight = 1.0
_C.expression.enable = 0
_C.expression.use_conf_weight = False
_C.expression.prior = CN()
_C.expression.prior.type = 'l2'
_C.expression.prior.weight = 0.0
_C.expression.prior.margin = 1.0
_C.expression.prior.use_vector = True
_C.expression.prior.norm = 'l2'
_C.expression.prior.barrier = 'log'
_C.expression.prior.epsilon = 1e-7

_C.global_orient = CN()
_C.global_orient.type = 'rotation'
_C.global_orient.enable = 0
_C.global_orient.weight = 1.0
_C.global_orient.prior = CN()

_C.body_pose = CN()
_C.body_pose.type = 'rotation'
_C.body_pose.enable = 0
_C.body_pose.weight = 1.0
_C.body_pose.prior = CN()
_C.body_pose.prior.type = 'l2'
_C.body_pose.prior.use_max = False
_C.body_pose.prior.weight = 0.0
_C.body_pose.prior.path = 'data/priors/gmm_08.pkl'
_C.body_pose.prior.num_gaussians = 8

_C.left_hand_pose = CN()
_C.left_hand_pose.use_conf_weight = False
_C.left_hand_pose.type = 'rotation'
_C.left_hand_pose.enable = 0
_C.left_hand_pose.weight = 1.0
_C.left_hand_pose.prior = CN()
_C.left_hand_pose.prior.type = 'l2'
_C.left_hand_pose.prior.weight = 0.0
_C.left_hand_pose.prior.num_gaussians = 6
_C.left_hand_pose.prior.path = 'data/priors/gmm_left_06.pkl'

_C.right_hand_pose = CN()
_C.right_hand_pose.use_conf_weight = False
_C.right_hand_pose.type = 'rotation'
_C.right_hand_pose.enable = 0
_C.right_hand_pose.weight = 1.0
_C.right_hand_pose.prior = CN()
_C.right_hand_pose.prior.type = 'l2'
_C.right_hand_pose.prior.weight = 0.0
_C.right_hand_pose.prior.num_gaussians = 6
_C.right_hand_pose.prior.path = 'data/priors/gmm_right_06.pkl'

_C.jaw_pose = CN()
_C.jaw_pose.type = 'rotation'
_C.jaw_pose.use_conf_weight = False
_C.jaw_pose.enable = 0
_C.jaw_pose.weight = 1.0
_C.jaw_pose.prior = CN()
_C.jaw_pose.prior.type = 'l2'
_C.jaw_pose.prior.weight = 0.0
_C.jaw_pose.prior.reduction = 'mean'

_C.edge = CN()
_C.edge.weight = 0.0
_C.edge.type = 'vertex-edge'
_C.edge.norm_type = 'l2'
_C.edge.gt_edge_path = ''
_C.edge.est_edge_path = ''
_C.edge.rho = 100.0
_C.edge.size_average = True
_C.edge.enable = 0

_C.hand = CN()

_C.hand.joints_2d = CN()
_C.hand.joints_2d.weight = 1.0
_C.hand.joints_2d.type = 'keypoints'
_C.hand.joints_2d.norm_type = 'l1'
_C.hand.joints_2d.robustifier = 'none'
_C.hand.joints_2d.rho = 100.0
_C.hand.joints_2d.beta = 5.0 / 100 * 2
_C.hand.joints_2d.size_average = True
_C.hand.joints_2d.enable = 0

_C.hand.vertices = CN()
_C.hand.vertices.weight = 0.0
_C.hand.vertices.type = 'weighted-l1'
_C.hand.vertices.rho = 100.0
_C.hand.vertices.beta = 5.0 / 100 * 2
_C.hand.vertices.size_average = True
_C.hand.vertices.enable = 0

_C.hand.edge = CN()
_C.hand.edge.weight = 0.0
_C.hand.edge.type = 'vertex-edge'
_C.hand.edge.norm_type = 'l2'
_C.hand.edge.gt_edge_path = ''
_C.hand.edge.est_edge_path = ''
_C.hand.edge.rho = 100.0
_C.hand.edge.size_average = True
_C.hand.edge.enable = 0

_C.hand.hand_edge_2d = CN()
_C.hand.hand_edge_2d.weight = 0.0
_C.hand.hand_edge_2d.norm_type = 'l2'
_C.hand.hand_edge_2d.rho = 100.0
_C.hand.hand_edge_2d.beta = 5.0 / 100 * 2
_C.hand.hand_edge_2d.size_average = True
_C.hand.hand_edge_2d.enable = 0
_C.hand.hand_edge_2d.robustifier = 'none'
_C.hand.hand_edge_2d.scale = 1.0
_C.hand.hand_edge_2d.threshold = 1.0


_C.hand.joints_3d = CN()
_C.hand.joints_3d.weight = 0.0
_C.hand.joints_3d.type = 'keypoints'
_C.hand.joints_3d.norm_type = 'l1'
_C.hand.joints_3d.rho = 100.0
_C.hand.joints_3d.beta = 5.0 / 100 * 2
_C.hand.joints_3d.size_average = True
_C.hand.joints_3d.enable = 500 * 1000


_C.hand.shape = CN()
_C.hand.shape.type = 'l2'
_C.hand.shape.weight = 0.0
_C.hand.shape.enable = 0
_C.hand.shape.prior = CN()
_C.hand.shape.prior.weight = 0.0
_C.hand.shape.prior.type = 'l2'
_C.hand.shape.prior.margin = 1.0
_C.hand.shape.prior.norm = 'l2'
_C.hand.shape.prior.use_vector = True
_C.hand.shape.prior.barrier = 'log'
_C.hand.shape.prior.epsilon = 1e-7

_C.hand.global_orient = CN()
_C.hand.global_orient.type = 'rotation'
_C.hand.global_orient.enable = 0
_C.hand.global_orient.weight = 1.0
_C.hand.global_orient.prior = CN()

_C.hand.hand_pose = CN()
_C.hand.hand_pose.use_conf_weight = False
_C.hand.hand_pose.type = 'rotation'
_C.hand.hand_pose.enable = 0
_C.hand.hand_pose.weight = 1.0
_C.hand.hand_pose.prior = CN()
_C.hand.hand_pose.prior.type = 'l2'
_C.hand.hand_pose.prior.weight = 0.0
_C.hand.hand_pose.prior.num_gaussians = 6
_C.hand.hand_pose.prior.margin = 1.0
_C.hand.hand_pose.prior.path = 'data/priors/gmm_left_06.pkl'

# Losses
_C.head = CN()

_C.head.joints_2d = CN()
_C.head.joints_2d.type = 'keypoints'
_C.head.joints_2d.norm_type = 'l1'
_C.head.joints_2d.robustifier = 'none'
_C.head.joints_2d.rho = 100.0
_C.head.joints_2d.beta = 5.0 / 100 * 2
_C.head.joints_2d.size_average = True
_C.head.joints_2d.weight = 0.0
_C.head.joints_2d.enable = 0.0

_C.head.edge_2d = CN()
_C.head.edge_2d.weight = 0.0
_C.head.edge_2d.norm_type = 'l2'
_C.head.edge_2d.rho = 100.0
_C.head.edge_2d.beta = 5.0 / 100 * 2
_C.head.edge_2d.size_average = True
_C.head.edge_2d.enable = 0
_C.head.edge_2d.robustifier = 'none'
_C.head.edge_2d.scale = 0.0
_C.head.edge_2d.threshold = 1.0

_C.head.vertices = CN()
_C.head.vertices.weight = 0.0
_C.head.vertices.type = 'weighted-l1'
_C.head.vertices.rho = 100.0
_C.head.vertices.beta = 5.0 / 100 * 2
_C.head.vertices.size_average = True
_C.head.vertices.enable = 0

_C.head.edge = CN()
_C.head.edge.weight = 0.0
_C.head.edge.type = 'vertex-edge'
_C.head.edge.norm_type = 'l2'
_C.head.edge.gt_edge_path = ''
_C.head.edge.est_edge_path = ''
_C.head.edge.rho = 100.0
_C.head.edge.size_average = True
_C.head.edge.enable = 0

_C.head.joints_3d = CN()
_C.head.joints_3d.weight = 0.0
_C.head.joints_3d.type = 'keypoints'
_C.head.joints_3d.norm_type = 'l1'
_C.head.joints_3d.rho = 100.0
_C.head.joints_3d.beta = 5.0 / 100 * 2
_C.head.joints_3d.size_average = True
_C.head.joints_3d.enable = 0.0

_C.head.shape = CN()
_C.head.shape.type = 'l2'
_C.head.shape.weight = 1.0
_C.head.shape.enable = 0
_C.head.shape.prior = CN()
_C.head.shape.prior.type = 'l2'
_C.head.shape.prior.weight = 0.0
_C.head.shape.prior.margin = 1.0
_C.head.shape.prior.norm = 'l2'
_C.head.shape.prior.use_vector = True
_C.head.shape.prior.barrier = 'log'
_C.head.shape.prior.epsilon = 1e-7

_C.head.expression = CN()
_C.head.expression.type = 'l2'
_C.head.expression.weight = 1.0
_C.head.expression.enable = 0
_C.head.expression.use_conf_weight = False
_C.head.expression.prior = CN()
_C.head.expression.prior.type = 'l2'
_C.head.expression.prior.weight = 0.0
_C.head.expression.prior.margin = 1.0
_C.head.expression.prior.use_vector = True
_C.head.expression.prior.norm = 'l2'
_C.head.expression.prior.barrier = 'log'
_C.head.expression.prior.epsilon = 1e-7

_C.head.global_orient = CN()
_C.head.global_orient.type = 'rotation'
_C.head.global_orient.enable = 0
_C.head.global_orient.weight = 1.0
_C.head.global_orient.prior = CN()

_C.head.jaw_pose = CN()
_C.head.jaw_pose.type = 'rotation'
_C.head.jaw_pose.use_conf_weight = False
_C.head.jaw_pose.enable = 0
_C.head.jaw_pose.weight = 1.0
_C.head.jaw_pose.prior = CN()
_C.head.jaw_pose.prior.type = 'l2'
_C.head.jaw_pose.prior.weight = 0.0
