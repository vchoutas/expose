from fvcore.common.config import CfgNode as CN
#  from yacs.config import CfgNode as CN

_C = CN()

_C.body_model = CN()

_C.body_model.j14_regressor_path = ''
_C.body_model.mean_pose_path = ''
_C.body_model.shape_mean_path = 'data/shape_mean.npy'
_C.body_model.type = 'smplx'
_C.body_model.model_folder = 'models'
_C.body_model.use_compressed = True
_C.body_model.gender = 'neutral'
_C.body_model.num_betas = 10
_C.body_model.num_expression_coeffs = 10
_C.body_model.use_feet_keypoints = True
_C.body_model.use_face_keypoints = True
_C.body_model.use_face_contour = False

_C.body_model.global_orient = CN()
# The configuration for the parameterization of the body pose
_C.body_model.global_orient.param_type = 'cont_rot_repr'

_C.body_model.body_pose = CN()
# The configuration for the parameterization of the body pose
_C.body_model.body_pose.param_type = 'cont_rot_repr'
_C.body_model.body_pose.finetune = False

_C.body_model.left_hand_pose = CN()
# The configuration for the parameterization of the left hand pose
_C.body_model.left_hand_pose.param_type = 'pca'
_C.body_model.left_hand_pose.num_pca_comps = 12
_C.body_model.left_hand_pose.flat_hand_mean = False
# The type of prior on the left hand pose

_C.body_model.right_hand_pose = CN()
# The configuration for the parameterization of the left hand pose
_C.body_model.right_hand_pose.param_type = 'pca'
_C.body_model.right_hand_pose.num_pca_comps = 12
_C.body_model.right_hand_pose.flat_hand_mean = False

_C.body_model.jaw_pose = CN()
_C.body_model.jaw_pose.param_type = 'cont_rot_repr'
_C.body_model.jaw_pose.data_fn = 'clusters.pkl'

####### HAND MODEL ########

_C.hand_model = CN()
_C.hand_model.j14_regressor_path = ''
_C.hand_model.mean_pose_path = ''
_C.hand_model.shape_mean_path = 'data/shape_mean.npy'
_C.hand_model.type = 'mano-from-smplx'
_C.hand_model.model_folder = 'models'
_C.hand_model.use_compressed = True
_C.hand_model.gender = 'neutral'
_C.hand_model.num_betas = 10
_C.hand_model.num_expression_coeffs = 10
_C.hand_model.use_feet_keypoints = True
_C.hand_model.use_face_keypoints = True

_C.hand_model.return_hand_vertices_only = True
_C.hand_model.vertex_idxs_path = ''

_C.hand_model.global_orient = CN()
# The configuration for the parameterization of the body pose
_C.hand_model.global_orient.param_type = 'cont_rot_repr'

_C.hand_model.hand_pose = CN()
_C.hand_model.hand_pose.param_type = 'pca'
_C.hand_model.hand_pose.num_pca_comps = 12
_C.hand_model.hand_pose.flat_hand_mean = False

#### HEAD MODEL ###########
_C.head_model = CN()
_C.head_model.j14_regressor_path = ''
_C.head_model.mean_pose_path = ''
_C.head_model.shape_mean_path = 'data/shape_mean.npy'
_C.head_model.type = 'flame-from-smplx'
_C.head_model.model_folder = 'models'
_C.head_model.use_compressed = True
_C.head_model.gender = 'neutral'
_C.head_model.num_betas = 10
_C.head_model.num_expression_coeffs = 10
_C.head_model.use_feet_keypoints = True
_C.head_model.use_face_keypoints = True
_C.head_model.use_face_contour = True
_C.head_model.return_head_vertices_only = True
_C.head_model.vertex_idxs_path = ''

_C.head_model.global_orient = CN()
# The configuration for the parameterization of the body pose
_C.head_model.global_orient.param_type = 'cont_rot_repr'
#
_C.head_model.jaw_pose = CN()
_C.head_model.jaw_pose.param_type = 'cont_rot_repr'
