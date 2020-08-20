from copy import deepcopy
from fvcore.common.config import CfgNode as CN

_C = CN()

_C = CN()
_C.type = 'sgd'
_C.num_epochs = 300
_C.lr = 1e-4
_C.offsets_decay = 1e-4

_C.steps = (30000,)

_C.sgd = CN()
_C.sgd.momentum = 0.9
_C.sgd.nesterov = True

_C.scheduler = CN()
_C.scheduler.type = 'none'
_C.scheduler.gamma = 0.1
_C.scheduler.milestones = []
_C.scheduler.step_size = 1000
_C.scheduler.warmup_factor = 1.0e-1 / 3
_C.scheduler.warmup_iters = 500
_C.scheduler.warmup_method = "linear"

# Adam parameters
_C.adam = CN()
_C.adam.betas = [0.9, 0.999]
_C.adam.eps = 1e-08
_C.adam.amsgrad = False

_C.rmsprop = CN()
_C.rmsprop.alpha = 0.99

_C.weight_decay = 0.0
_C.weight_decay_bias = 0.0
_C.bias_lr_factor = 1.0
