import copy
import numpy as np # noqa
import math # noqa
from collections import OrderedDict

from miu.mmd.PmxData import PmxModel, Vertex, Material, Bone, Morph, DisplaySlot, RigidBody, Joint # noqa
from miu.mmd.VmdData import VmdMotion, VmdBoneFrame, VmdCameraFrame, VmdInfoIk, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame # noqa
from miu.module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from miu.module.MParams import BoneLinks # noqa
from miu.utils.MLogger import MLogger # noqa

logger = MLogger(__name__, level=1)

class VmdService:

    def __init__(self, joints: np.ndarray):
        self.joints = joints

    def execute():
        pass


    