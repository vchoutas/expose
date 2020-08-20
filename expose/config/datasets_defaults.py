#  from yacs.config import CfgNode as CN
from fvcore.common.config import CfgNode as CN

###############################################################################
############################## DATASETS ##############################


def build_transform_cfg(node, key='transforms', flip_prob=0.0,
                        downsample_factor_min=1.0,
                        downsample_factor_max=1.0,
                        center_jitter_factor=0.0,
                        downsample_dist='categorical',
                        ):
    if key not in node:
        node[key] = CN()
    node[key].flip_prob = flip_prob
    node[key].downsample_dist = downsample_dist
    node[key].downsample_factor_min = downsample_factor_min
    node[key].downsample_factor_max = downsample_factor_max
    node[key].downsample_cat_factors = (1.0,)
    node[key].center_jitter_factor = center_jitter_factor
    node[key].center_jitter_dist = 'normal'
    node[key].crop_size = 256
    node[key].scale_factor_min = 1.0
    node[key].scale_factor_max = 1.0
    node[key].scale_factor = 0.0
    node[key].scale_dist = 'uniform'
    node[key].noise_scale = 0.0
    node[key].rotation_factor = 0.0
    node[key].mean = [0.485, 0.456, 0.406]
    node[key].std = [0.229, 0.224, 0.225]
    node[key].brightness = 0.0
    node[key].saturation = 0.0
    node[key].hue = 0.0
    node[key].contrast = 0.0

    return node[key]


def build_num_workers_cfg(node, key='num_workers'):
    if key not in node:
        node[key] = CN()
    node[key].train = 8
    node[key].val = 2
    node[key].test = 2
    return node[key]


_C = CN()

_C.use_equal_sampling = True
_C.use_face_contour = False
_C.use_packed = False

_C.body = CN()

_C.body.vertex_flip_correspondences = ''

_C.use_hands = True
_C.use_face = True
_C.body.batch_size = 1

_C.body.splits = CN()
_C.body.splits.train = ['spin', 'curated_fits']
_C.body.splits.val = []
_C.body.splits.test = []

_C.body.num_workers = build_num_workers_cfg(_C.body)
_C.body.ratio_2d = 0.5
_C.body.transforms = build_transform_cfg(
    _C.body, key='transforms')

_C.body.openpose = CN()
_C.body.openpose.data_folder = 'data/openpose'
_C.body.openpose.img_folder = 'images'
_C.body.openpose.keyp_folder = 'keypoints'
_C.body.openpose.body_thresh = 0.05
_C.body.openpose.hand_thresh = 0.2
_C.body.openpose.head_thresh = 0.3
_C.body.openpose.keyp_format = 'coco25'
_C.body.openpose.use_face_contour = _C.use_face_contour

_C.body.tracks = CN()
_C.body.tracks.data_folder = 'data/tracks'
_C.body.tracks.img_folder = 'images'
_C.body.tracks.keyp_folder = 'keypoints'
_C.body.tracks.body_thresh = 0.05
_C.body.tracks.hand_thresh = 0.2
_C.body.tracks.head_thresh = 0.3
_C.body.tracks.keyp_format = 'coco25'
_C.body.tracks.use_face_contour = _C.use_face_contour

_C.body.ehf = CN()
_C.body.ehf.data_folder = 'data/EHF'
_C.body.ehf.img_folder = 'images'
_C.body.ehf.keyp_folder = 'keypoints'
_C.body.ehf.joints_to_ign = []
_C.body.ehf.alignments_folder = 'alignments'
_C.body.ehf.metrics = ['v2v']
_C.body.ehf.use_face_contour = _C.use_face_contour

_C.body.curated_fits = CN()
_C.body.curated_fits.data_path = 'data/curated_fits'
_C.body.curated_fits.img_folder = ''
_C.body.curated_fits.use_face_contour = _C.use_face_contour
_C.body.curated_fits.joints_to_ign = []
_C.body.curated_fits.metrics = ['v2v']
_C.body.curated_fits.body_thresh = 0.1
_C.body.curated_fits.hand_thresh = 0.2
_C.body.curated_fits.face_thresh = 0.4
_C.body.curated_fits.min_hand_keypoints = 8
_C.body.curated_fits.min_head_keypoints = 8
_C.body.curated_fits.binarization = True
_C.body.curated_fits.use_joint_conf = False
_C.body.curated_fits.return_params = True
_C.body.curated_fits.return_shape = True


_C.body.mpii = CN()
_C.body.mpii.data_path = ''
_C.body.mpii.img_folder = 'images'
_C.body.mpii.keyp_folder = 'keypoints'
_C.body.mpii.use_face_contour = _C.use_face_contour
_C.body.mpii.only_with_hands = True

_C.body.threedpw = CN()
_C.body.threedpw.data_path = 'data/3dpw'
_C.body.threedpw.img_folder = ''
_C.body.threedpw.param_folder = ''
_C.body.threedpw.seq_folder = 'sequenceFiles'
_C.body.threedpw.metrics = ['mpjpe14', 'v2v']
_C.body.threedpw.body_thresh = 0.05
_C.body.threedpw.binarization = True

_C.body.lsp_test = CN()
_C.body.lsp_test.data_path = 'data/lsp_test'
_C.body.lsp_test.use_face_contour = _C.use_face_contour

_C.body.spin = CN()
_C.body.spin.img_folder = 'data/spin/images'
_C.body.spin.npz_files = [
    'mpii.npz', 'lsp.npz', 'lspet.npz', 'coco.npz']
_C.body.spin.return_params = True
_C.body.spin.use_face_contour = False
_C.body.spin.binarization = True
_C.body.spin.body_thresh = 0.1
_C.body.spin.hand_thresh = 0.2
_C.body.spin.face_thresh = 0.4
_C.body.spin.min_hand_keypoints = 8
_C.body.spin.min_head_keypoints = 8
_C.body.spin.return_shape = False
_C.body.spin.return_full_pose = False
_C.body.spin.metrics = []


_C.body.spinx = CN()
_C.body.spinx.img_folder = 'data/spinx/images'
_C.body.spinx.vertex_folder = 'data/spinx/vertices'
_C.body.spinx.npz_files = [
    'mpii.npz', 'lsp.npz', 'lspet.npz', 'coco.npz']
_C.body.spinx.return_params = True
_C.body.spinx.use_face_contour = False
_C.body.spinx.binarization = True
_C.body.spinx.body_thresh = 0.1
_C.body.spinx.hand_thresh = 0.2
_C.body.spinx.face_thresh = 0.4
_C.body.spinx.min_hand_keypoints = 8
_C.body.spinx.min_head_keypoints = 8
_C.body.spinx.return_shape = False
_C.body.spinx.return_full_pose = False
_C.body.spinx.return_expression = False
_C.body.spinx.return_vertices = True
_C.body.spinx.metrics = []


_C.hand = CN()

_C.hand.vertex_flip_correspondences = ''
_C.hand.splits = CN()
_C.hand.splits.train = ['freihand']
_C.hand.splits.val = ['freihand']
_C.hand.splits.test = []
_C.hand.batch_size = 32

_C.hand.num_workers = build_num_workers_cfg(_C.hand)
_C.hand.ratio_2d = 0.5
_C.hand.scale_factor = 1.2
_C.hand.transforms = build_transform_cfg(
    _C.hand, key='transforms')

_C.hand.openpose = CN()
_C.hand.openpose.data_folder = 'data/openpose'
_C.hand.openpose.img_folder = 'images'
_C.hand.openpose.keyp_folder = 'keypoints'
_C.hand.openpose.body_thresh = 0.05
_C.hand.openpose.hand_thresh = 0.2
_C.hand.openpose.head_thresh = 0.3
_C.hand.openpose.is_right = False
_C.hand.openpose.keyp_format = 'coco25'

_C.hand.freihand = CN()
_C.hand.freihand.data_path = 'data/freihand'
_C.hand.freihand.metrics = ['mpjpe', 'v2v']
_C.hand.freihand.return_vertices = True
_C.hand.freihand.return_params = True
_C.hand.freihand.file_format = 'npz'

_C.hand.ehf = CN()
_C.hand.ehf.data_folder = 'data/EHF'
_C.hand.ehf.img_folder = 'images'
_C.hand.ehf.keyp_folder = 'keypoints'
_C.hand.ehf.joints_to_ign = []
_C.hand.ehf.alignments_folder = 'alignments'
_C.hand.ehf.metrics = ['v2v']
_C.hand.ehf.use_face_contour = _C.use_face_contour
_C.hand.ehf.is_right = False

_C.hand.curated_fits = CN()
_C.hand.curated_fits.data_path = 'data/curated_fits'
_C.hand.curated_fits.img_folder = ''
_C.hand.curated_fits.use_face_contour = _C.use_face_contour
_C.hand.curated_fits.joints_to_ign = []
_C.hand.curated_fits.metrics = ['v2v']
_C.hand.curated_fits.body_thresh = 0.1
_C.hand.curated_fits.hand_thresh = 0.2
_C.hand.curated_fits.face_thresh = 0.4
_C.hand.curated_fits.binarization = True
_C.hand.curated_fits.use_joint_conf = False
_C.hand.curated_fits.return_params = True
_C.hand.curated_fits.return_shape = True

_C.hand.spinx = CN()
_C.hand.spinx.img_folder = 'data/spinx/images'
_C.hand.spinx.npz_files = ['mpii.npz', 'lsp.npz', 'lspet.npz',
                           'coco.npz']
_C.hand.spinx.return_params = True
_C.hand.spinx.use_face_contour = False
_C.hand.spinx.binarization = True
_C.hand.spinx.body_thresh = 0.1
_C.hand.spinx.hand_thresh = 0.2
_C.hand.spinx.face_thresh = 0.4
_C.hand.spinx.return_shape = False
_C.hand.spinx.return_full_pose = False
_C.hand.spinx.metrics = []


_C.head = CN()

_C.head.vertex_flip_correspondences = ''
_C.head.splits = CN()
_C.head.splits.train = ['ffhq']
_C.head.splits.val = ['ffhq']
_C.head.splits.test = []
_C.head.batch_size = 32

_C.head.num_workers = build_num_workers_cfg(_C.head)
_C.head.ratio_2d = 0.5
_C.head.scale_factor = 1.2
_C.head.transforms = build_transform_cfg(
    _C.head, key='transforms')

_C.head.ffhq = CN()
_C.head.ffhq.data_path = 'data/ffhq'
_C.head.ffhq.img_folder = 'images'
_C.head.ffhq.param_fname = 'ffhq_parameters_with_keyps.npz'
_C.head.ffhq.use_face_contour = True
_C.head.ffhq.return_vertices = False
_C.head.ffhq.split_size = 0.95
_C.head.ffhq.metrics = ['v2v']

_C.head.stirling3d = CN()
_C.head.stirling3d.data_path = 'data/stirling/HQ'

_C.head.openpose = CN()
_C.head.openpose.data_folder = 'data/openpose'
_C.head.openpose.img_folder = 'images'
_C.head.openpose.keyp_folder = 'keypoints'
_C.head.openpose.body_thresh = 0.05
_C.head.openpose.hand_thresh = 0.2
_C.head.openpose.head_thresh = 0.3
_C.head.openpose.keyp_format = 'coco25'

_C.head.ehf = CN()
_C.head.ehf.data_folder = 'data/EHF'
_C.head.ehf.img_folder = 'images'
_C.head.ehf.keyp_folder = 'keypoints'
_C.head.ehf.joints_to_ign = []
_C.head.ehf.alignments_folder = 'alignments'
_C.head.ehf.metrics = ['v2v']
_C.head.ehf.use_face_contour = _C.use_face_contour

# TODO: Add handhe-Wild datasets
_C.head.curated_fits = CN()
_C.head.curated_fits.data_path = 'data/curated_fits'
_C.head.curated_fits.img_folder = ''
_C.head.curated_fits.use_face_contour = _C.use_face_contour
_C.head.curated_fits.joints_to_ign = []
_C.head.curated_fits.metrics = ['v2v', 'mpjpe14', 'keyp2d_error']
_C.head.curated_fits.body_thresh = 0.1
_C.head.curated_fits.hand_thresh = 0.2
_C.head.curated_fits.face_thresh = 0.4
_C.head.curated_fits.binarization = True
_C.head.curated_fits.use_joint_conf = False
_C.head.curated_fits.return_params = True
_C.head.curated_fits.return_shape = True

_C.head.spinx = CN()
_C.head.spinx.img_folder = 'data/spinx/images'
_C.head.spinx.npz_files = ['mpii.npz', 'lsp.npz', 'lspet.npz',
                           'coco.npz']
_C.head.spinx.return_params = True
_C.head.spinx.use_face_contour = False
_C.head.spinx.binarization = True
_C.head.spinx.body_thresh = 0.1
_C.head.spinx.hand_thresh = 0.2
_C.head.spinx.face_thresh = 0.4
_C.head.spinx.return_shape = False
_C.head.spinx.return_full_pose = False
_C.head.spinx.metrics = []
