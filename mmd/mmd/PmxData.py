# -*- coding: utf-8 -*-
#
import _pickle as cPickle
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import math
import numpy as np

from mmd.module.MParams import BoneLinks
from mmd.module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from mmd.utils.MException import SizingException # noqa
from mmd.utils.MLogger import MLogger # noqa

logger = MLogger(__name__, level=MLogger.DEBUG)


class Deform:
    def __init__(self, index0):
        self.index0 = index0

class Bdef1(Deform):
    def __init__(self, index0):
        self.index0 = index0
    
    def get_idx_list(self):
        return [self.index0]
        
    def __str__(self):
        return "<Bdef1 {0}>".format(self.index0)

class Bdef2(Deform):
    def __init__(self, index0, index1, weight0):
        self.index0 = index0
        self.index1 = index1
        self.weight0 = weight0
        
    def get_idx_list(self):
        return [self.index0, self.index1]
        
    def __str__(self):
        return "<Bdef2 {0}, {1}, {2}>".format(self.index0, self.index1, self.weight0)

class Bdef4(Deform):
    def __init__(self, index0, index1, index2, index3, weight0, weight1, weight2, weight3):
        self.index0 = index0
        self.index1 = index1
        self.index2 = index2
        self.index3 = index3
        self.weight0 = weight0
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        
    def get_idx_list(self):
        return [self.index0, self.index1, self.index2, self.index3]

    def __str__(self):
        return "<Bdef4 {0}:{1}, {2}:{3}, {4}:{5}, {6}:{7}>".format(self.index0, self.index1, self.index2, self.index3, self.weight0, self.weight1, self.weight2, self.weight3)
            
class Sdef(Deform):
    def __init__(self, index0, index1, weight0, sdef_c, sdef_r0, sdef_r1):
        self.index0 = index0
        self.index1 = index1
        self.weight0 = weight0
        self.sdef_c = sdef_c
        self.sdef_r0 = sdef_r0
        self.sdef_r1 = sdef_r1
        
    def get_idx_list(self):
        return [self.index0, self.index1]

    def __str__(self):
        return "<Sdef {0}, {1}, {2}, {3} {4} {5}>".format(self.index0, self.index1, self.weight0, self.sdef_c, self.sdef_r0, self.sdef_r1)
    
class Qdef(Deform):
    def __init__(self, index0, index1, weight0, sdef_c, sdef_r0, sdef_r1):
        self.index0 = index0
        self.index1 = index1
        self.weight0 = weight0
        self.sdef_c = sdef_c
        self.sdef_r0 = sdef_r0
        self.sdef_r1 = sdef_r1
        
    def get_idx_list(self):
        return [self.index0, self.index1]

    def __str__(self):
        return "<Sdef {0}, {1}, {2}, {3} {4} {5}>".format(self.index0, self.index1, self.weight0, self.sdef_c, self.sdef_r0, self.sdef_r1)


# 頂点構造 ----------------------------
class Vertex:

    def __init__(self, index, position, normal, uv, extended_uvs, deform, edge_factor):
        self.index = index
        self.position = position
        self.normal = normal
        self.uv = uv
        self.extended_uvs = extended_uvs or []
        self.deform = deform
        self.edge_factor = edge_factor
        
    def __str__(self):
        return "<Vertex index:{0}, position:{1}, normal:{2}, uv:{3}, extended_uv: {4}, deform:{5}, edge:{6}".format(
               self.index, self.position, self.normal, self.uv, len(self.extended_uvs), self.deform, self.edge_factor)

    def is_deform_index(self, target_idx):
        if type(self.deform) is Bdef1:
            return self.deform.index0 == target_idx
        elif type(self.deform) is Bdef2:
            return self.deform.index0 == target_idx or self.deform.index1 == target_idx
        elif type(self.deform) is Bdef4:
            return self.deform.index0 == target_idx or self.deform.index1 == target_idx \
                or self.deform.index2 == target_idx or self.deform.index3 == target_idx
        elif type(self.deform) is Sdef:
            return self.deform.index0 == target_idx or self.deform.index1 == target_idx
        elif type(self.deform) is Qdef:
            return self.deform.index0 == target_idx or self.deform.index1 == target_idx

        return False
    
    # 最もウェイトが乗っているボーンINDEX
    def get_max_deform_index(self, head_links_indexes):
        if type(self.deform) is Bdef2 or type(self.deform) is Sdef or type(self.deform) is Qdef:
            if self.deform.weight0 >= 0.5 and self.deform.index0 in head_links_indexes.keys():
                return self.deform.index0
            else:
                if self.deform.index1 in head_links_indexes.keys():
                    return self.deform.index1
                else:
                    return self.deform.index0

        elif type(self.deform) is Bdef4:
            
            # 上半身系INDEXにウェイトが乗っているボーンのみ対象
            target_weights = []
            if self.deform.index0 in head_links_indexes.keys():
                target_weights.append(self.deform.weight0)
            if self.deform.index1 in head_links_indexes.keys():
                target_weights.append(self.deform.weight1)
            if self.deform.index2 in head_links_indexes.keys():
                target_weights.append(self.deform.weight2)
            if self.deform.index3 in head_links_indexes.keys():
                target_weights.append(self.deform.weight3)
                    
            max_weight = max(target_weights)

            if max_weight == self.deform.weight1:
                return self.deform.index1
            elif max_weight == self.deform.weight2:
                return self.deform.index2
            elif max_weight == self.deform.weight3:
                return self.deform.index3
            else:
                return self.deform.index0

        return self.deform.index0
    


# 材質構造-----------------------
class Material:
    def __init__(self, name, english_name, diffuse_color, alpha, specular_factor, specular_color, ambient_color, flag, edge_color, edge_size, texture_index,
                 sphere_texture_index, sphere_mode, toon_sharing_flag, toon_texture_index=0, comment="", vertex_count=0):
        self.name = name
        self.english_name = english_name
        self.diffuse_color = diffuse_color
        self.alpha = alpha
        self.specular_color = specular_color
        self.specular_factor = specular_factor
        self.ambient_color = ambient_color
        self.flag = flag
        self.edge_color = edge_color
        self.edge_size = edge_size
        self.texture_index = texture_index
        self.sphere_texture_index = sphere_texture_index
        self.sphere_mode = sphere_mode
        self.toon_sharing_flag = toon_sharing_flag
        self.toon_texture_index = toon_texture_index
        self.comment = comment
        self.vertex_count = vertex_count

    def __str__(self):
        return "<Material name:{0}, english_name:{1}, diffuse_color:{2}, alpha:{3}, specular_color:{4}, " \
               "ambient_color: {5}, flag: {6}, edge_color: {7}, edge_size: {8}, texture_index: {9}, " \
               "sphere_texture_index: {10}, sphere_mode: {11}, toon_sharing_flag: {12}, " \
               "toon_texture_index: {13}, comment: {14}, vertex_count: {15}".format(
                   self.name, self.english_name, self.diffuse_color, self.alpha, self.specular_color,
                   self.ambient_color, self.flag, self.edge_color, self.edge_size, self.texture_index,
                   self.sphere_texture_index, self.sphere_mode, self.toon_sharing_flag,
                   self.toon_texture_index, self.comment, self.vertex_count)

class Ik:
    def __init__(self, target_index, loop, limit_radian, link=None):
        self.target_index = target_index
        self.loop = loop
        self.limit_radian = limit_radian
        self.link = link or []

    def __str__(self):
        return "<Ik target_index:{0}, loop:{1}, limit_radian:{2}, link:{3}".format(self.target_index, self.loop, self.limit_radian, self.link)
        
class IkLink:

    def __init__(self, bone_index, limit_angle, limit_min=None, limit_max=None):
        self.bone_index = bone_index
        self.limit_angle = limit_angle
        self.limit_min = limit_min or MVector3D()
        self.limit_max = limit_max or MVector3D()

    def __str__(self):
        return "<IkLink bone_index:{0}, limit_angle:{1}, limit_min:{2}, limit_max:{3}".format(self.bone_index, self.limit_angle, self.limit_min, self.limit_max)
        
# ボーン構造-----------------------
class Bone:
    def __init__(self, name, english_name, position, parent_index, layer, flag, tail_position=None, tail_index=-1, effect_index=-1, effect_factor=0.0, fixed_axis=None,
                 local_x_vector=None, local_z_vector=None, external_key=-1, ik=None):
        self.name = name
        self.english_name = english_name
        self.position = position
        self.parent_index = parent_index
        self.layer = layer
        self.flag = flag
        self.tail_position = tail_position or MVector3D()
        self.tail_index = tail_index
        self.effect_index = effect_index
        self.effect_factor = effect_factor
        self.fixed_axis = fixed_axis or MVector3D()
        self.local_x_vector = local_x_vector or MVector3D()
        self.local_z_vector = local_z_vector or MVector3D()
        self.external_key = external_key
        self.ik = ik
        self.index = -1
        # 表示枠チェック時にONにするので、デフォルトはFalse
        self.display = False

        # 親ボーンからの長さ3D版(計算して求める）
        self.len_3d = MVector3D()
        # オフセット(ローカル)
        self.local_offset = MVector3D()
        # IKオフセット(グローバル)
        self.global_ik_offset = MVector3D()
        
        # IK制限角度
        self.ik_limit_min = MVector3D()
        self.ik_limit_max = MVector3D()
        # IK内積上限値
        self.dot_limit = 0
        # IK内積上限値(近接)
        self.dot_near_limit = 0
        # IK内積上限値(遠目)
        self.dot_far_limit = 0
        # IK内積上限値(単体)
        self.dot_single_limit = 0
        # IK単位角度
        self.degree_limit = 360

        self.BONEFLAG_TAILPOS_IS_BONE = 0x0001
        self.BONEFLAG_CAN_ROTATE = 0x0002
        self.BONEFLAG_CAN_TRANSLATE = 0x0004
        self.BONEFLAG_IS_VISIBLE = 0x0008
        self.BONEFLAG_CAN_MANIPULATE = 0x0010
        self.BONEFLAG_IS_IK = 0x0020
        self.BONEFLAG_IS_EXTERNAL_ROTATION = 0x0100
        self.BONEFLAG_IS_EXTERNAL_TRANSLATION = 0x0200
        self.BONEFLAG_HAS_FIXED_AXIS = 0x0400
        self.BONEFLAG_HAS_LOCAL_COORDINATE = 0x0800
        self.BONEFLAG_IS_AFTER_PHYSICS_DEFORM = 0x1000
        self.BONEFLAG_IS_EXTERNAL_PARENT_DEFORM = 0x2000
    
    def copy(self):
        return cPickle.loads(cPickle.dumps(self, -1))

    def hasFlag(self, flag):
        return (self.flag & flag) != 0

    def setFlag(self, flag, enable):
        if enable:
            self.flag |= flag
        else:
            self.flag &= ~flag

    def getConnectionFlag(self):
        return self.hasFlag(self.BONEFLAG_TAILPOS_IS_BONE)

    def getRotatable(self):
        return self.hasFlag(self.BONEFLAG_CAN_ROTATE)

    def getTranslatable(self):
        return self.hasFlag(self.BONEFLAG_CAN_TRANSLATE)

    def getVisibleFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_VISIBLE)

    def getManipulatable(self):
        return self.hasFlag(self.BONEFLAG_CAN_MANIPULATE)

    def getIkFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_IK)

    def getExternalRotationFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_EXTERNAL_ROTATION)

    def getExternalTranslationFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_EXTERNAL_TRANSLATION)

    def getFixedAxisFlag(self):
        return self.hasFlag(self.BONEFLAG_HAS_FIXED_AXIS)

    def getLocalCoordinateFlag(self):
        return self.hasFlag(self.BONEFLAG_HAS_LOCAL_COORDINATE)

    def getAfterPhysicsDeformFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_AFTER_PHYSICS_DEFORM)

    def getExternalParentDeformFlag(self):
        return self.hasFlag(self.BONEFLAG_IS_EXTERNAL_PARENT_DEFORM)

    def __str__(self):
        return "<Bone name:{0}, english_name:{1}, position:{2}, parent_index:{3}, layer:{4}, " \
               "flag: {5}, tail_position: {6}, tail_index: {7}, effect_index: {8}, effect_factor: {9}, " \
               "fixed_axis: {10}, local_x_vector: {11}, local_z_vector: {12}, " \
               "external_key: {13}, ik: {14}, index: {15}".format(
                   self.name, self.english_name, self.position, self.parent_index, self.layer,
                   self.flag, self.tail_position, self.tail_index, self.effect_index, self.effect_factor,
                   self.fixed_axis, self.local_x_vector, self.local_z_vector,
                   self.external_key, self.ik, self.index)


# モーフ構造-----------------------
class Morph:
    def __init__(self, name, english_name, panel, morph_type, offsets=None):
        self.index = 0
        self.name = name
        self.english_name = english_name
        self.panel = panel
        self.morph_type = morph_type
        self.offsets = offsets or []
        # 表示枠チェック時にONにするので、デフォルトはFalse
        self.display = False
        self.related_names = []

    def __str__(self):
        return "<Morph name:{0}, english_name:{1}, panel:{2}, morph_type:{3}, offsets(len): {4}".format(
               self.name, self.english_name, self.panel, self.morph_type, len(self.offsets))
    
    # パネルの名称取得
    def get_panel_name(self):
        if self.panel == 1:
            return "眉"
        elif self.panel == 2:
            return "目"
        elif self.panel == 3:
            return "口"
        elif self.panel == 4:
            return "他"
        else:
            return "？"
            
    class GroupMorphData:
        def __init__(self, morph_index, value):
            self.morph_index = morph_index
            self.value = value

    class VertexMorphOffset:
        def __init__(self, vertex_index, position_offset):
            self.vertex_index = vertex_index
            self.position_offset = position_offset

    class BoneMorphData:
        def __init__(self, bone_index, position, rotation):
            self.bone_index = bone_index
            self.position = position
            self.rotation = rotation

    class UVMorphData:
        def __init__(self, vertex_index, uv):
            self.vertex_index = vertex_index
            self.uv = uv

    class MaterialMorphData:
        def __init__(self, material_index, calc_mode, diffuse, specular, specular_factor, ambient, edge_color, edge_size, texture_factor, sphere_texture_factor, toon_texture_factor):
            self.material_index = material_index
            self.calc_mode = calc_mode
            self.diffuse = diffuse
            self.specular = specular
            self.specular_factor = specular_factor
            self.ambient = ambient
            self.edge_color = edge_color
            self.edge_size = edge_size
            self.texture_factor = texture_factor
            self.sphere_texture_factor = sphere_texture_factor
            self.toon_texture_factor = toon_texture_factor


# 表示枠構造-----------------------
class DisplaySlot:
    def __init__(self, name, english_name, special_flag, references=None):
        self.name = name
        self.english_name = english_name
        self.special_flag = special_flag
        self.references = references or []

    def __str__(self):
        return "<DisplaySlots name:{0}, english_name:{1}, special_flag:{2}, references(len):{3}".format(self.name, self.english_name, self.special_flag, len(self.references))


# 剛体構造-----------------------
class RigidBody:
    def __init__(self, name, english_name, bone_index, collision_group, no_collision_group, shape_type, shape_size, shape_position, shape_rotation, mass, linear_damping, \
                 angular_damping, restitution, friction, mode):
        self.name = name
        self.english_name = english_name
        self.bone_index = bone_index
        self.collision_group = collision_group
        self.no_collision_group = no_collision_group
        self.shape_type = shape_type
        self.shape_size = shape_size
        self.shape_position = shape_position
        self.shape_rotation = shape_rotation
        self.param = RigidBodyParam(mass, linear_damping, angular_damping, restitution, friction)
        self.mode = mode
        self.index = -1
        self.bone_name = ""
        self.is_arm_upper = False
        self.is_small = False

        self.SHAPE_SPHERE = 0
        self.SHAPE_BOX = 1
        self.SHAPE_CAPSULE = 2

    def __str__(self):
        return "<RigidBody name:{0}, english_name:{1}, bone_index:{2}, collision_group:{3}, no_collision_group:{4}, " \
               "shape_type: {5}, shape_size: {6}, shape_position: {7}, shape_rotation: {8}, param: {9}, " \
               "mode: {10}".format(self.name, self.english_name, self.bone_index, self.collision_group, self.no_collision_group,
                                   self.shape_type, self.shape_size, self.shape_position.to_log(), self.shape_rotation.to_log(), self.param, self.mode)
    
    # 剛体: ボーン追従
    def isModeStatic(self):
        return self.mode == 0
    
    # 剛体: 物理演算
    def isModeDynamic(self):
        return self.mode == 1
    
    # 剛体: 物理演算 + Bone位置合わせ
    def isModeMix(self):
        return self.mode == 2
    
    def get_obb(self, fno, bone_pos, bone_matrix, is_aliginment, is_arm_left):
        # 剛体の形状別の衝突判定用
        if self.shape_type == self.SHAPE_SPHERE:
            return Sphere(fno, self.shape_size, self.shape_position, self.shape_rotation, self.bone_name, bone_pos, bone_matrix, is_aliginment, \
                                    is_arm_left, self.is_arm_upper, self.is_small, False)
        elif self.shape_type == self.SHAPE_BOX:
            return Box(fno, self.shape_size, self.shape_position, self.shape_rotation, self.bone_name, bone_pos, bone_matrix, is_aliginment, \
                                 is_arm_left, self.is_arm_upper, self.is_small, True)
        else:
            return Capsule(fno, self.shape_size, self.shape_position, self.shape_rotation, self.bone_name, bone_pos, bone_matrix, is_aliginment, \
                                     is_arm_left, self.is_arm_upper, self.is_small, True)

class RigidBodyParam:
    def __init__(self, mass, linear_damping, angular_damping, restitution, friction):
        self.mass = mass
        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.restitution = restitution
        self.friction = friction

    def __str__(self):
        return "<RigidBodyParam mass:{0}, linear_damping:{1}, angular_damping:{2}, restitution:{3}, friction: {4}".format(
            self.mass, self.linear_damping, self.angular_damping, self.restitution, self.friction)
            
# OBB（有向境界ボックス：Oriented Bounding Box）
class OBB:
    def __init__(self, fno, shape_size, shape_position, shape_rotation, bone_name, bone_pos, bone_matrix, is_aliginment, is_arm_left, is_arm_upper, is_small, is_init_rot):
        self.fno = fno
        self.shape_size = shape_size
        self.shape_position = shape_position
        self.shape_rotation = shape_rotation
        self.shape_rotation_qq = MQuaternion.fromEulerAngles(math.degrees(shape_rotation.x()), math.degrees(shape_rotation.y()), math.degrees(shape_rotation.z()))
        self.bone_pos = bone_pos
        self.h_sign = 1 if is_arm_left else -1
        self.v_sign = -1 if is_arm_upper and is_small else 1
        self.is_aliginment = is_aliginment
        self.is_arm_upper = is_arm_upper
        self.is_small = is_small
        self.is_arm_left = is_arm_left

        # 回転なし行列
        self.matrix = bone_matrix[bone_name].copy()
        # 回転あり行列
        self.rotated_matrix = bone_matrix[bone_name].copy()

        # 剛体自体の位置
        self.matrix.translate(self.shape_position - bone_pos)
        self.rotated_matrix.translate(self.shape_position - bone_pos)
        # 剛体自体の回転(回転用行列だけ保持)
        self.rotated_matrix.rotate(self.shape_rotation_qq)

        # 剛体自体の原点
        self.origin = self.matrix * MVector3D(0, 0, 0)

        self.origin_xyz = {"x": self.origin.x(), "y": self.origin.y(), "z": self.origin.z()}
        self.shape_size_xyz = {"x": self.shape_size.x(), "y": self.shape_size.y(), "z": self.shape_size.z()}

    # OBBとの衝突判定
    def get_collistion(self, point, root_global_pos, max_length):
        pass
    
# 球剛体
class Sphere(OBB):
    def __init__(self, *args):
        super().__init__(*args)

    # 衝突しているか
    def get_collistion(self, point, root_global_pos, max_length):

        # 原点との距離が半径未満なら衝突
        d = point.distanceToPoint(self.origin)
        collision = 0 < d < self.shape_size.x() * 0.98
        near_collision = 0 <= d <= self.shape_size.x() * 1.02

        x_distance = 0
        z_distance = 0
        rep_x_collision_vec = MVector3D()
        rep_z_collision_vec = MVector3D()

        if collision or near_collision:
            # 剛体のローカル座標系に基づく点の位置
            local_point = self.matrix.inverted() * point

            x = self.shape_size.x() * 1.02 * self.h_sign
            y = self.shape_size.x() * 1.02 * self.v_sign
            z = self.shape_size.x() * 1.02 * -1  # (np.sign(local_point.z()) if self.is_arm_upper else -1)

            # 各軸方向の離れ具合
            x_theta = math.acos(max(-1, min(1, local_point.x() / x)))
            y_theta = math.acos(max(-1, min(1, local_point.y() / y)))
            z_theta = math.acos(max(-1, min(1, local_point.z() / z)))
            # 離れ具合から見た円周の位置
            sin_y_theta = math.sin(y_theta) * 1.02
            sin_x_theta = math.sin(x_theta) * 1.02
            sin_z_theta = math.sin(z_theta) * 1.02

            new_y = local_point.y()

            new_x_local = MVector3D(y_theta * x, new_y, local_point.z())
            new_z_local = MVector3D(local_point.x(), new_y, y_theta * z)

            x_distance = new_x_local.distanceToPoint(local_point)
            z_distance = new_z_local.distanceToPoint(local_point)

            rep_x_collision_vec = self.matrix * new_x_local
            rep_z_collision_vec = self.matrix * new_z_local

            # 腕の位置を起点とする行列（移動量だけ見る）
            arm_matrix = MMatrix4x4()
            arm_matrix.setToIdentity()
            arm_matrix.translate(root_global_pos)

            # 腕から見た回避位置
            x_arm_local = arm_matrix.inverted() * rep_x_collision_vec
            z_arm_local = arm_matrix.inverted() * rep_z_collision_vec

            if x_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                x_arm_local *= (max_length / x_arm_local.length()) * 0.98
                rep_x_collision_vec = arm_matrix * x_arm_local
                new_x_local = self.matrix.inverted() * rep_x_collision_vec
                x_distance = new_x_local.distanceToPoint(local_point)

            if z_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                z_arm_local *= (max_length / z_arm_local.length()) * 0.98
                rep_z_collision_vec = arm_matrix * z_arm_local
                new_z_local = self.matrix.inverted() * rep_z_collision_vec
                z_distance = new_z_local.distanceToPoint(local_point)

            logger.debug("f: %s, y: %s, yt: %s, sy: %s, xt: %s, sx: %s, zt: %s, sz: %s, xd: %s, zd: %s, l: %s, d: %s, xl: %s, zl: %s, xr: %s, zr: %s", \
                            self.fno, local_point.y() / y, y_theta, sin_y_theta, x_theta, sin_x_theta, z_theta, sin_z_theta, x_distance, z_distance, local_point.to_log(), d, \
                            new_x_local.to_log(), new_z_local.to_log(), rep_x_collision_vec, rep_z_collision_vec)

        # 3方向の間に点が含まれていたら衝突あり
        return (collision, near_collision, x_distance, z_distance, rep_x_collision_vec, rep_z_collision_vec)

# 箱剛体
class Box(OBB):
    def __init__(self, *args):
        super().__init__(*args)

    # 衝突しているか（内外判定）
    # https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
    def get_collistion(self, point, root_global_pos, max_length):
        # 立方体の中にある場合、衝突

        # ---------
        # 下辺
        b1 = self.matrix * MVector3D(-self.shape_size.x(), -self.shape_size.y(), -self.shape_size.z())
        b2 = self.matrix * MVector3D(self.shape_size.x(), -self.shape_size.y(), -self.shape_size.z())
        b4 = self.matrix * MVector3D(-self.shape_size.x(), -self.shape_size.y(), self.shape_size.z())
        # 上辺
        t1 = self.matrix * MVector3D(-self.shape_size.x(), self.shape_size.y(), -self.shape_size.z())

        d1 = (t1 - b1)
        size1 = d1.length()
        dir1 = d1 / size1
        dir1.effective()

        d2 = (b2 - b1)
        size2 = d2.length()
        dir2 = d2 / size2
        dir2.effective()

        d3 = (b4 - b1)
        size3 = d3.length()
        dir3 = d3 / size3
        dir3.effective()

        dir_vec = point - self.origin
        dir_vec.effective()

        res1 = abs(MVector3D.dotProduct(dir_vec, dir1)) * 2 < size1
        res2 = abs(MVector3D.dotProduct(dir_vec, dir2)) * 2 < size2
        res3 = abs(MVector3D.dotProduct(dir_vec, dir3)) * 2 < size3

        # 3方向の間に点が含まれていたら衝突あり
        collision = (res1 and res2 and res3 and True)

        # ---------
        # 下辺
        b1 = self.matrix * MVector3D(-self.shape_size.x(), -self.shape_size.y(), -self.shape_size.z()) * 1.02
        b2 = self.matrix * MVector3D(self.shape_size.x(), -self.shape_size.y(), -self.shape_size.z()) * 1.02
        b4 = self.matrix * MVector3D(-self.shape_size.x(), -self.shape_size.y(), self.shape_size.z()) * 1.02
        # 上辺
        t1 = self.matrix * MVector3D(-self.shape_size.x(), self.shape_size.y(), -self.shape_size.z()) * 1.02

        d1 = (t1 - b1)
        size1 = d1.length()
        dir1 = d1 / size1
        dir1.effective()

        d2 = (b2 - b1)
        size2 = d2.length()
        dir2 = d2 / size2
        dir2.effective()

        d3 = (b4 - b1)
        size3 = d3.length()
        dir3 = d3 / size3
        dir3.effective()

        dir_vec = point - self.origin
        dir_vec.effective()

        res1 = abs(MVector3D.dotProduct(dir_vec, dir1)) * 2 < size1
        res2 = abs(MVector3D.dotProduct(dir_vec, dir2)) * 2 < size2
        res3 = abs(MVector3D.dotProduct(dir_vec, dir3)) * 2 < size3

        # 3方向の間に点が含まれていたら衝突あり
        near_collision = (res1 and res2 and res3 and True)

        x_distance = 0
        z_distance = 0
        rep_x_collision_vec = MVector3D()
        rep_z_collision_vec = MVector3D()

        if collision or near_collision:
            # 左右の腕のどちらと衝突しているかにより、元に戻す方向が逆になる
            x = self.shape_size.x() * 1.02 * self.h_sign
            z = -self.shape_size.z() * 1.02
            
            # X方向にOBBの境界に持って行った場合の位置
            x_base = self.rotated_matrix * MVector3D(x, 0, 0)
            # Z方向に同上
            z_base = self.rotated_matrix * MVector3D(0, 0, z)
            logger.test("x_base: %s", x_base)
            logger.test("z_base: %s", z_base)

            x_diff = x_base.distanceToPoint(point)
            z_diff = z_base.distanceToPoint(point)

            logger.test("x_diff: %s", x_diff)
            logger.test("z_diff: %s", z_diff)

            # 剛体のローカル座標系に基づく点の位置
            local_point = self.rotated_matrix.inverted() * point

            new_y = local_point.y()

            new_x_local = MVector3D(x, new_y, local_point.z())
            new_z_local = MVector3D(local_point.x(), new_y, z)

            x_distance = new_x_local.distanceToPoint(local_point)
            z_distance = new_z_local.distanceToPoint(local_point)

            rep_x_collision_vec = self.rotated_matrix * new_x_local
            rep_z_collision_vec = self.rotated_matrix * new_z_local

            # 腕の位置を起点とする行列（移動量だけ見る）
            arm_matrix = MMatrix4x4()
            arm_matrix.setToIdentity()
            arm_matrix.translate(root_global_pos)

            # 腕から見た回避位置
            x_arm_local = arm_matrix.inverted() * rep_x_collision_vec
            z_arm_local = arm_matrix.inverted() * rep_z_collision_vec

            if x_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                x_arm_local *= (max_length / x_arm_local.length()) * 0.98
                rep_x_collision_vec = arm_matrix * x_arm_local
                new_x_local = self.matrix.inverted() * rep_x_collision_vec
                x_distance = new_x_local.distanceToPoint(local_point)

            if z_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                z_arm_local *= (max_length / z_arm_local.length()) * 0.98
                rep_z_collision_vec = arm_matrix * z_arm_local
                new_z_local = self.matrix.inverted() * rep_z_collision_vec
                z_distance = new_z_local.distanceToPoint(local_point)

            logger.debug("f: %s, xd: %s, zd: %s, l: %s, xl: %s, zl: %s, xr: %s, zr: %s", \
                            self.fno, x_distance, z_distance, local_point.to_log(), new_x_local.to_log(), new_z_local.to_log(), rep_x_collision_vec, rep_z_collision_vec)

        return (collision, near_collision, x_distance, z_distance, rep_x_collision_vec, rep_z_collision_vec)

# カプセル剛体
class Capsule(OBB):
    def __init__(self, *args):
        super().__init__(*args)

    # 衝突しているか
    # http://marupeke296.com/COL_3D_No27_CapsuleCapsule.html
    def get_collistion(self, point, root_global_pos, max_length):

        # 下辺
        b1 = self.rotated_matrix * MVector3D(0, -self.shape_size.y(), 0)
        # 上辺
        t1 = self.rotated_matrix * MVector3D(0, self.shape_size.y(), 0)

        # 垂線までの長さ
        v = (t1 - b1)
        lensq = v.lengthSquared()
        t = 0 if lensq == 0 else MVector3D.dotProduct(v, point - b1) / lensq
        # 垂線を下ろした座標
        h = b1 + (v * t)

        logger.test("v: %s", v)
        logger.test("lensq: %s", lensq)
        logger.test("t: %s", t)
        logger.test("h: %s", h)

        # 点・下辺始点・垂線点の三角形
        ba = (point - b1).lengthSquared()
        bb = (h - b1).lengthSquared()
        bc = (point - h).lengthSquared()

        # 点・上辺終点・垂線点の三角形
        ta = (point - t1).lengthSquared()
        tb = (h - t1).lengthSquared()
        tc = (point - h).lengthSquared()

        logger.test("ba: %s, bb: %s, bc: %s", ba, bb, bc)
        logger.test("ta: %s, tb: %s, tc: %s", ta, tb, tc)

        if t1.distanceToPoint(b1) < b1.distanceToPoint(h) < t1.distanceToPoint(h):
            # b1側の外分点
            h = b1
        elif t1.distanceToPoint(b1) < t1.distanceToPoint(h) < b1.distanceToPoint(h):
            # t1側の外分点
            h = t1

        logger.test("v: %s", v)
        logger.test("lensq: %s", lensq)
        logger.test("t: %s", t)
        logger.test("h: %s", h)
        logger.test("point: %s", point)
        logger.test("segl: %s", point.distanceToPoint(h))

        # カプセルの線分から半径以内なら中に入っている
        d = point.distanceToPoint(h)
        collision = 0 < d < self.shape_size.x() * 0.98
        near_collision = 0 <= d <= self.shape_size.x() * 1.02

        x_distance = 0
        z_distance = 0
        rep_x_collision_vec = MVector3D()
        rep_z_collision_vec = MVector3D()

        if collision or near_collision:
            # hのローカル座標系に基づく点の位置
            h_matrix = self.matrix.copy()
            h_matrix.translate(self.matrix.inverted() * h)
            local_point = h_matrix.inverted() * point
            logger.debug("h: %s, localh: %s", h, h_matrix * MVector3D())

            # 距離分だけ離した場合の球
            x = d * 1.02 * self.h_sign
            y = d * 1.02 * self.v_sign
            z = d * 1.02 * -1    # (np.sign(local_point.z()) if self.is_arm_upper else -1)

            # 各軸方向の離れ具合
            x_theta = math.acos(max(-1, min(1, local_point.x() / x)))
            y_theta = math.acos(max(-1, min(1, abs(local_point.y()) / y)))
            z_theta = math.acos(max(-1, min(1, local_point.z() / z)))
            # 離れ具合から見た円周の位置
            sin_y_theta = math.sin(y_theta) * 1.02
            sin_x_theta = math.sin(x_theta) * 1.02
            sin_z_theta = math.sin(z_theta) * 1.02

            new_y = local_point.y()

            new_x_local = MVector3D(y_theta * x, new_y, local_point.z())
            new_z_local = MVector3D(local_point.x(), new_y, y_theta * z)

            x_distance = new_x_local.distanceToPoint(local_point)
            z_distance = new_z_local.distanceToPoint(local_point)

            rep_x_collision_vec = h_matrix * new_x_local
            rep_z_collision_vec = h_matrix * new_z_local

            # 腕の位置を起点とする行列（移動量だけ見る）
            arm_matrix = MMatrix4x4()
            arm_matrix.setToIdentity()
            arm_matrix.translate(root_global_pos)

            # 腕から見た回避位置
            x_arm_local = arm_matrix.inverted() * rep_x_collision_vec
            z_arm_local = arm_matrix.inverted() * rep_z_collision_vec

            if x_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                x_arm_local *= (max_length / x_arm_local.length()) * 0.98
                rep_x_collision_vec = arm_matrix * x_arm_local
                new_x_local = h_matrix.inverted() * rep_x_collision_vec
                x_distance = new_x_local.distanceToPoint(local_point)

            if z_arm_local.length() >= max_length:
                # 最大可能距離より長い場合、縮める
                z_arm_local *= (max_length / z_arm_local.length()) * 0.98
                rep_z_collision_vec = arm_matrix * z_arm_local
                new_z_local = h_matrix.inverted() * rep_z_collision_vec
                z_distance = new_z_local.distanceToPoint(local_point)

            logger.debug("f: %s, localy: %s, y_theta: %s, sin_y_theta: %s, x_theta: %s, sin_x_theta: %s, z_theta: %s, sin_z_theta: %s, x_distance: %s, z_distance: %s, "\
                            "local_point: [%s], d: %s, new_x_local: %s, new_z_local: %s, rep_x_collision_vec: %s, rep_z_collision_vec: %s", \
                            self.fno, local_point.y() / y, y_theta, sin_y_theta, x_theta, sin_x_theta, z_theta, sin_z_theta, x_distance, z_distance, local_point.to_log(), d, \
                            new_x_local.to_log(), new_z_local.to_log(), rep_x_collision_vec, rep_z_collision_vec)

        # 3方向の間に点が含まれていたら衝突あり
        return (collision, near_collision, x_distance, z_distance, rep_x_collision_vec, rep_z_collision_vec)


# ジョイント構造-----------------------
class Joint:
    def __init__(self, name, english_name, joint_type, rigidbody_index_a, rigidbody_index_b, position, rotation, \
                 translation_limit_min, translation_limit_max, rotation_limit_min, rotation_limit_max, spring_constant_translation, spring_constant_rotation):
        self.name = name
        self.english_name = english_name
        self.joint_type = joint_type
        self.rigidbody_index_a = rigidbody_index_a
        self.rigidbody_index_b = rigidbody_index_b
        self.position = position
        self.rotation = rotation
        self.translation_limit_min = translation_limit_min
        self.translation_limit_max = translation_limit_max
        self.rotation_limit_min = rotation_limit_min
        self.rotation_limit_max = rotation_limit_max
        self.spring_constant_translation = spring_constant_translation
        self.spring_constant_rotation = spring_constant_rotation

    def __str__(self):
        return "<Joint name:{0}, english_name:{1}, joint_type:{2}, rigidbody_index_a:{3}, rigidbody_index_b:{4}, " \
               "position: {5}, rotation: {6}, translation_limit_min: {7}, translation_limit_max: {8}, " \
               "spring_constant_translation: {9}, spring_constant_rotation: {10}".format(
                   self.name, self.english_name, self.joint_type, self.rigidbody_index_a, self.rigidbody_index_b,
                   self.position, self.rotation, self.translation_limit_min, self.translation_limit_max,
                   self.spring_constant_translation, self.spring_constant_rotation)


class PmxModel:
    def __init__(self):
        self.path = ''
        self.name = ''
        self.english_name = ''
        self.comment = ''
        self.english_comment = ''
        # 頂点データ（キー：ボーンINDEX、値：頂点データリスト）
        self.vertices = {}
        # 面データ
        self.indices = []
        # テクスチャデータ
        self.textures = []
        # 材質データ
        self.materials = {}
        # 材質データ（キー：材質INDEX、値：材質名）
        self.material_indexes = {}
        # ボーンデータ
        self.bones = {}
        # ボーンINDEXデータ（キー：ボーンINDEX、値：ボーン名）
        self.bone_indexes = {}
        # モーフデータ(順番保持)
        self.morphs = {}
        # 表示枠データ
        self.display_slots = {}
        # 剛体データ
        self.rigidbodies = {}
        # 剛体INDEXデータ
        self.rigidbody_indexes = {}
        # ジョイントデータ
        self.joints = {}
        # ハッシュ値
        self.digest = None
        # 上半身がサイジング可能（標準・準標準ボーン構造）か
        self.can_upper_sizing = True
        # 腕がサイジング可能（標準・準標準ボーン構造）か
        self.can_arm_sizing = True
        # 頭頂頂点
        self.head_top_vertex = None
        # 左足底頂点
        self.left_sole_vertex = None
        # 右足底頂点
        self.right_sole_vertex = None
        # 左つま先頂点
        self.left_toe_vertex = None
        # 右つま先頂点
        self.right_toe_vertex = None
        # 左右手のひら頂点
        self.wrist_entity_vertex = {}
        # 左右ひじ頂点
        self.elbow_entity_vertex = {}
        # 左右ひじ手首中間頂点
        self.elbow_middle_entity_vertex = {}
    
    # ローカルX軸の取得
    def get_local_x_axis(self, bone_name: str):
        if bone_name not in self.bones:
            return MVector3D()
        
        bone = self.bones[bone_name]
        to_pos = MVector3D()

        if bone.fixed_axis != MVector3D():
            # 軸制限がある場合、親からの向きを保持
            fixed_x_axis = bone.fixed_axis.normalized()
        else:
            fixed_x_axis = MVector3D()
        
        from_pos = self.bones[bone.name].position
        if bone.tail_position != MVector3D():
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        elif bone.tail_index >= 0 and bone.tail_index in self.bone_indexes and self.bones[self.bone_indexes[bone.tail_index]].position != bone.position:
            # 表示先が指定されているの場合、保持
            to_pos = self.bones[self.bone_indexes[bone.tail_index]].position
        else:
            # 表示先がない場合、とりあえず子ボーンのどれかを選択
            for b in self.bones.values():
                if b.parent_index == bone.index and self.bones[self.bone_indexes[b.index]].position != bone.position:
                    to_pos = self.bones[self.bone_indexes[b.index]].position
                    break
        
        # 軸制限の指定が無い場合、子の方向
        x_axis = (to_pos - from_pos).normalized()

        if fixed_x_axis != MVector3D() and np.sign(fixed_x_axis.x()) != np.sign(x_axis.x()):
            # 軸制限の軸方向と計算上の軸方向が違う場合、逆ベクトル
            x_axis = -fixed_x_axis

        return x_axis
    
    # 腕のスタンスの違い
    def calc_arm_stance(self, from_bone_name: str, to_bone_name=None):
        default_pos = MVector3D(1, 0, 0) if "左" in from_bone_name else MVector3D(-1, 0, 0)
        return self.calc_stance(from_bone_name, to_bone_name, default_pos)

    # 指定ボーン間のスタンス
    def calc_stance(self, from_bone_name: str, to_bone_name: str, default_pos: MVector3D):
        from_pos = MVector3D()
        to_pos = MVector3D()

        if from_bone_name in self.bones:
            fv = self.bones[from_bone_name]
            from_pos = fv.position
            
            if to_bone_name in self.bones:
                # TOが指定されている場合、ボーン位置を保持
                to_pos = self.bones[to_bone_name].position
            else:
                # TOの明示が無い場合、表示先からボーン位置取得
                if fv.tail_position != MVector3D():
                    # 表示先が相対パスの場合、保持
                    to_pos = from_pos + fv.tail_position
                elif fv.tail_index >= 0 and fv.tail_index in self.bone_indexes:
                    # 表示先がボーンの場合、ボーン位置保持
                    to_pos = self.bones[self.bone_indexes[fv.tail_index]].position
                else:
                    # ここまで来たらデフォルト加算
                    to_pos = from_pos + default_pos

        from_qq = MQuaternion()
        diff_pos = MVector3D()
        if from_pos != MVector3D() and to_pos != MVector3D():
            logger.test("from_pos: %s", from_pos)
            logger.test("to_pos: %s", to_pos)

            diff_pos = to_pos - from_pos
            diff_pos.normalize()
            logger.test("diff_pos: %s", diff_pos)

            from_qq = MQuaternion.rotationTo(default_pos, diff_pos)
            logger.test("[z] from_bone_name: %s, from_qq: %s", from_bone_name, from_qq.toEulerAngles())

        return diff_pos, from_qq

    # 腕系サイジングが可能かチェック
    def check_arm_bone_can_sizing(self):

        target_bones = ["左腕", "左ひじ", "左手首", "右腕", "右ひじ", "右手首"]

        cannot_sizing = "腕系処理をスキップします。\n腕系処理（腕スタンス補正・捩り分散・接触回避・位置合わせ）を実行したい場合、\n腕タブのチェックスキップFLGをONにして再実行してください。"

        if not set(target_bones).issubset(self.bones.keys()):
            logger.warning("腕・ひじ・手首の左右ボーンが揃ってないため、%s\nモデル: %s", cannot_sizing, self.name, decoration=MLogger.DECORATION_BOX)
            return False
        
        for bone_name in self.bones.keys():
            if ("腕IK" in bone_name or "腕ＩＫ" in bone_name or "うでIK" in bone_name or "うでＩＫ" in bone_name or "腕XIK" in bone_name):
                # 腕IKが入ってて、かつそれが表示されてる場合、NG
                logger.warning("モデルに「腕IK」に類するボーンが含まれているため、%s\nモデル: %s", cannot_sizing, self.name, decoration=MLogger.DECORATION_BOX)
                return False

        return True
    
    # ボーンリンク生成
    def create_link_2_top_lr(self, *target_bone_types, **kwargs):
        is_defined = kwargs["is_defined"] if "is_defined" in kwargs else True

        for target_bone_type in target_bone_types:
            left_links = self.create_link_2_top_one("左{0}".format(target_bone_type), is_defined=is_defined)
            right_links = self.create_link_2_top_one("右{0}".format(target_bone_type), is_defined=is_defined)

            if left_links and right_links:
                # IKリンクがある場合、そのまま返す
                return {"左": left_links, "右": right_links}

    # ボーンリンク生成
    def create_link_2_top_one(self, *target_bone_names, **kwargs):
        is_defined = kwargs["is_defined"] if "is_defined" in kwargs else True

        for target_bone_name in target_bone_names:
            links = self.create_link_2_top(target_bone_name, None, is_defined)

            if links and target_bone_name in links.all():
                reversed_links = BoneLinks()
                
                # リンクがある場合、反転させて返す
                for lname in reversed(list(links.all().keys())):
                    reversed_links.append(links.get(lname))

                return reversed_links
        
        # 最後まで回しても取れなかった場合、エラー
        raise SizingException("ボーンリンクの生成に失敗しました。モデル「%s」に「%s」のボーンがあるか確認してください。" % (self.name, ",".join(target_bone_names)))

    # リンク生成
    def create_link_2_top(self, target_bone_name: str, links: BoneLinks, is_defined: bool):
        if not links:
            # まだリンクが生成されていない場合、順序保持辞書生成
            links = BoneLinks()
        
        if target_bone_name not in self.bones and target_bone_name not in self.PARENT_BORN_PAIR:
            # 開始ボーン名がなければ終了
            return links

        start_type_bone = target_bone_name
        if target_bone_name.startswith("右") or target_bone_name.startswith("左"):
            # 左右から始まってたらそれは除く
            start_type_bone = target_bone_name[1:]

        # 自分をリンクに登録
        links.append(self.bones[target_bone_name].copy())

        parent_name = None
        if is_defined:
            # 定義済みの場合
            if target_bone_name not in self.PARENT_BORN_PAIR:
                raise SizingException("ボーンリンクの生成に失敗しました。モデル「%s」の「%s」ボーンが準標準までの構造ではない可能性があります。" % (self.name, target_bone_name))
                
            for pname in self.PARENT_BORN_PAIR[target_bone_name]:
                # 親子関係のボーンリストから親ボーンが存在した場合
                if pname in self.bones:
                    parent_name = pname
                    break
        else:
            # 未定義でよい場合
            if self.bones[target_bone_name].parent_index >= 0:
                # 親ボーンが存在している場合
                parent_name = self.bone_indexes[self.bones[target_bone_name].parent_index]

        if not parent_name:
            # 親ボーンがボーンインデックスリストになければ終了
            return links
        
        logger.test("target_bone_name: %s. parent_name: %s, start_type_bone: %s", target_bone_name, parent_name, start_type_bone)
        
        # 親をたどる
        try:
            return self.create_link_2_top(parent_name, links, is_defined)
        except RecursionError:
            raise SizingException("ボーンリンクの生成に失敗しました。\nモデル「{0}」の「{1}」ボーンで以下を確認してください。\n" \
                                  + "・同じ名前のボーンが複数ないか（ボーンのINDEXがズレるため、サイジングに失敗します）\n" \
                                  + "・親ボーンに自分の名前と同じ名前のボーンが指定されていないか\n※ PMXEditorの「PMXデータの状態検証」から確認できます。".format(self.name, target_bone_name))
    
    # 子孫ボーンリスト取得
    def get_child_bones(self, target_bone: Bone, bone_list=None):
        if not bone_list:
            bone_list = []
        
        child_bone_list = []

        for child_bone in self.bones.values():
            if child_bone.index != target_bone.index and child_bone.parent_index == target_bone.index:
                # 処理対象ボーンが親INDEXに入ってる場合、処理対象
                bone_list.append(child_bone)
                child_bone_list.append(child_bone)

        for child_bone in child_bone_list:
            self.get_child_bones(child_bone, bone_list)

        return bone_list

    # ボーン関係親子のペア
    PARENT_BORN_PAIR = {
        "SIZING_ROOT_BONE": [""],
        "全ての親": ["SIZING_ROOT_BONE"],
        "センター": ["全ての親", "SIZING_ROOT_BONE"],
        "グルーブ": ["センター"],
        "センター実体": ["グルーブ", "センター"],
        "腰": ["センター実体", "グルーブ", "センター"],
        "足中間": ["下半身"],
        "下半身": ["腰", "センター実体", "グルーブ", "センター"],
        "上半身": ["腰", "センター実体", "グルーブ", "センター"],
        "上半身2": ["上半身"],
        "首根元": ["上半身2", "上半身"],
        "首根元2": ["首根元", "上半身2", "上半身"],
        "首": ["首根元2", "首根元", "上半身2", "上半身"],
        "頭": ["首"],
        "頭頂実体": ["頭"],
        "左肩P": ["首根元2", "首根元", "上半身2", "上半身"],
        "左肩": ["左肩P", "首根元2", "首根元", "上半身2", "上半身"],
        "左肩下延長": ["左肩"],
        "左肩C": ["左肩"],
        "左腕": ["左肩C", "左肩"],
        "左腕捩": ["左腕"],
        "左腕ひじ中間": ["左腕捩", "左腕"],
        "左ひじ": ["左腕捩", "左腕"],
        "左ひじ実体": ["左ひじ"],
        "左手捩": ["左ひじ"],
        "左ひじ手首中間": ["左手捩", "左ひじ"],
        "左ひじ手首中間実体": ["左ひじ手首中間"],
        "左手首": ["左手捩", "左ひじ"],
        "左手首実体": ["左手首"],
        "左親指０": ["左手首"],
        "左親指１": ["左親指０", "左手首"],
        "左親指２": ["左親指１"],
        "左親指先実体": ["左親指２"],
        "左人指０": ["左手首"],
        "左人指１": ["左人指０", "左手首"],
        "左人指２": ["左人指１"],
        "左人指３": ["左人指２"],
        "左人指先実体": ["左人指３"],
        "左中指０": ["左手首"],
        "左中指１": ["左中指０", "左手首"],
        "左中指２": ["左中指１"],
        "左中指３": ["左中指２"],
        "左中指先実体": ["左中指３"],
        "左薬指０": ["左手首"],
        "左薬指１": ["左薬指０", "左手首"],
        "左薬指２": ["左薬指１"],
        "左薬指３": ["左薬指２"],
        "左薬指先実体": ["左薬指３"],
        "左小指０": ["左手首"],
        "左小指１": ["左小指０", "左手首"],
        "左小指２": ["左小指１"],
        "左小指３": ["左小指２"],
        "左小指先実体": ["左小指３"],
        "左足": ["足中間", "下半身"],
        "左ひざ": ["左足"],
        "左足首": ["左ひざ"],
        "左つま先": ["左足首"],
        "左足IK親": ["全ての親", "SIZING_ROOT_BONE"],
        "左足IK親底実体": ["左足IK親"],
        "左足ＩＫ": ["左足IK親", "全ての親", "SIZING_ROOT_BONE"],
        "左つま先ＩＫ": ["左足ＩＫ"],
        "左足ＩＫ底実体": ["左足ＩＫ"],
        "左足先EX": ["左つま先ＩＫ", "左足ＩＫ"],
        "左足底実体": ["左足先EX", "左つま先ＩＫ", "左足ＩＫ"],
        "左つま先実体": ["左足底実体", "左足先EX", "左つま先ＩＫ", "左足ＩＫ"],
        "右肩P": ["首根元2", "首根元", "上半身2", "上半身"],
        "右肩": ["右肩P", "首根元2", "首根元", "上半身2", "上半身"],
        "右肩下延長": ["右肩"],
        "右肩C": ["右肩"],
        "右腕": ["右肩C", "右肩"],
        "右腕捩": ["右腕"],
        "右腕ひじ中間": ["右腕捩", "右腕"],
        "右ひじ": ["右腕捩", "右腕"],
        "右ひじ実体": ["右ひじ"],
        "右手捩": ["右ひじ"],
        "右ひじ手首中間": ["右手捩", "右ひじ"],
        "右ひじ手首中間実体": ["右ひじ手首中間"],
        "右手首": ["右手捩", "右ひじ"],
        "右手首実体": ["右手首"],
        "右親指０": ["右手首"],
        "右親指１": ["右親指０", "右手首"],
        "右親指２": ["右親指１"],
        "右親指先実体": ["右親指２"],
        "右人指０": ["右手首"],
        "右人指１": ["右人指０", "右手首"],
        "右人指２": ["右人指１"],
        "右人指３": ["右人指２"],
        "右人指先実体": ["右人指３"],
        "右中指０": ["右手首"],
        "右中指１": ["右中指０", "右手首"],
        "右中指２": ["右中指１"],
        "右中指３": ["右中指２"],
        "右中指先実体": ["右中指３"],
        "右薬指０": ["右手首"],
        "右薬指１": ["右薬指０", "右手首"],
        "右薬指２": ["右薬指１"],
        "右薬指３": ["右薬指２"],
        "右薬指先実体": ["右薬指３"],
        "右小指０": ["右手首"],
        "右小指１": ["右小指０", "右手首"],
        "右小指２": ["右小指１"],
        "右小指３": ["右小指２"],
        "右小指先実体": ["右小指３"],
        "右足": ["足中間", "下半身"],
        "右ひざ": ["右足"],
        "右足首": ["右ひざ"],
        "右つま先": ["右足首"],
        "右足IK親": ["全ての親", "SIZING_ROOT_BONE"],
        "右足IK親底実体": ["右足IK親"],
        "右足ＩＫ": ["右足IK親", "全ての親", "SIZING_ROOT_BONE"],
        "右つま先ＩＫ": ["右足ＩＫ"],
        "右足ＩＫ底実体": ["右足ＩＫ"],
        "右足先EX": ["右つま先ＩＫ", "右足ＩＫ"],
        "右足底実体": ["右足先EX", "右つま先ＩＫ", "右足ＩＫ"],
        "右つま先実体": ["右足底実体", "右足先EX", "右つま先ＩＫ", "右足ＩＫ"],
        "左目": ["頭"],
        "右目": ["頭"]
    }
    
    # 頭頂の頂点を取得
    def get_head_top_vertex(self):
        bone_name_list = ["頭"]

        # まずX制限をかけて頂点を取得する
        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_original, def_is_target=self.def_is_target_x_limit)

        if not up_max_vertex:

            # 頭頂頂点が取れなかった場合、X制限を外す
            up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
                back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
                = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_original, def_is_target=None)

            if not up_max_vertex:
                if "頭" in self.bones:
                    return Vertex(-1, self.bones["頭"].position.copy(), MVector3D(), [], [], Bdef1(-1), -1)
                elif "首" in self.bones:
                    return Vertex(-1, self.bones["首"].position.copy(), MVector3D(), [], [], Bdef1(-1), -1)
                else:
                    return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        return up_max_vertex
    
    # 頭用剛体生成
    def get_head_rigidbody(self):
        bone_name_list = ["頭"]

        # 制限なしで前後左右上下
        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_original, def_is_target=None)

        if up_max_vertex and down_max_vertex and right_max_vertex and left_max_vertex and back_max_vertex and front_max_vertex:
            # Yは念のため首ボーンまで
            y_bottom = max(down_max_vertex.position.y(), self.bones["首"].position.y())
            # 頂点が取れた場合、半径算出
            # y_len = abs(up_max_vertex.position.y() - y_bottom)
            x_len = abs(left_max_vertex.position.x() - right_max_vertex.position.x())
            # z_len = abs(back_max_vertex.position.z() - front_max_vertex.position.z())

            # X頂点同士の長さの半分
            radius = x_len / 2

            # center = MVector3D()
            # # Yは下辺から半径分上
            # center.setY(down_max_vertex.position.y() + (radius * 0.95))
            # # Zは前面から半径分後ろ
            # center.setZ(front_max_vertex.position.z() + (radius * 0.95))

            # 中点
            center = MVector3D(np.mean([up_max_vertex.position.data(), down_max_vertex.position.data(), left_max_vertex.position.data(), \
                                        right_max_vertex.position.data(), back_max_vertex.position.data(), front_max_vertex.position.data()], axis=0))
            # Xはど真ん中
            center.setX(0)
            # Yは下辺からちょっと上
            center.setY(y_bottom + (radius * 0.98))
            # Zはちょっと前に
            center.setZ(center.z() - (radius * 0.05))

            head_rigidbody = RigidBody("頭接触回避", None, self.bones["頭"].index, 0, 0, 0, \
                                       MVector3D(radius, radius, radius), center, MVector3D(), 0, 0, 0, 0, 0, 0)
            head_rigidbody.bone_name = "頭"
            head_rigidbody.is_arm_upper = True

            return head_rigidbody
        
        return None

    # つま先の頂点を取得
    def get_toe_vertex(self, direction: str):
        # 足首より下で、指ではないボーン
        bone_name_list = []

        target_bone_name = None
        if "{0}足首".format(direction) in self.bones:
            target_bone_name = "{0}足首".format(direction)
        elif "{0}足ＩＫ".format(direction) in self.bones:
            target_bone_name = "{0}足ＩＫ".format(direction)
        else:
            # 足末端系ボーンがない場合、処理終了
            return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)

        # 足末端系ボーン
        for bk, bv in self.bones.items():
            if ((direction == "右" and bv.position.x() < 0) or (direction == "左" and bv.position.x() > 0)) \
               and bv.position.y() <= self.bones[target_bone_name].position.y() and direction in bk:
                bone_name_list.append(bk)
        
        if len(bone_name_list) == 0:
            # ウェイトボーンがない場合、つま先ボーン系の位置
            if "{0}つま先".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}つま先".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}つま先ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}つま先ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}足首".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足首".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}足ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            else:
                return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)

        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_original, def_is_target=None, \
                                       def_is_multi_target=self.def_is_multi_target_down_front, multi_target_default_val=MVector3D(0, 99999, 99999))

        if not front_max_vertex:
            # つま先頂点が取れなかった場合
            if "{0}つま先".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}つま先".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}つま先ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}つま先ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}足首".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足首".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            elif "{0}足ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            else:
                return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        return front_max_vertex

    # 足底の頂点を取得
    def get_sole_vertex(self, direction: str):
        # 足首より下で、指ではないボーン
        bone_name_list = []

        target_bone_name = None
        if "{0}足首".format(direction) in self.bones:
            target_bone_name = "{0}足首".format(direction)
        elif "{0}足ＩＫ".format(direction) in self.bones:
            target_bone_name = "{0}足ＩＫ".format(direction)
        else:
            # 足末端系ボーンがない場合、処理終了
            return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)

        # 足末端系ボーン
        for bk, bv in self.bones.items():
            if ((direction == "右" and bv.position.x() < 0) or (direction == "左" and bv.position.x() > 0)) \
               and bv.position.y() <= self.bones[target_bone_name].position.y() and direction in bk:
                bone_name_list.append(bk)
        
        if len(bone_name_list) == 0:
            # ウェイトボーンがない場合、足ＩＫの位置
            if "{0}足ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            else:
                return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)

        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_original, def_is_target=None, \
                                       def_is_multi_target=self.def_is_multi_target_down_front_sole, multi_target_default_val=MVector3D(0, 99999, 99999))

        if not multi_max_vertex:
            # 足底頂点が取れなかった場合
            if "{0}足ＩＫ".format(direction) in self.bones:
                return Vertex(-1, self.bones["{0}足ＩＫ".format(direction)].position, MVector3D(), [], [], Bdef1(-1), -1)
            else:
                return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        return multi_max_vertex

    # 手のひらの厚みをはかる頂点を取得
    def get_wrist_vertex(self, direction: str):
        # 足首より下で、指ではないボーン
        bone_name_list = []

        if "{0}手首".format(direction) in self.bones:
            for bk, bv in self.bones.items():
                if "{0}手首".format(direction) == bk:
                    bone_name_list.append(bk)
        else:
            # 手首ボーンがない場合、処理終了
            return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        # 腕の傾き（正確にはひじ以降の傾き）
        _, arm_stance_qq = self.calc_arm_stance("{0}ひじ".format(direction), "{0}手首".format(direction))

        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_horizonal, def_is_target=self.def_is_target_x_limit, \
                                       def_is_multi_target=self.def_is_multi_target_down_front, multi_target_default_val=MVector3D(0, 99999, 99999), qq4calc=arm_stance_qq)

        if not down_max_vertex:
            # 手首の下（手のひらの厚み）が取れなかった場合、X制限なしに取得する

            up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
                back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
                = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_horizonal, def_is_target=None, \
                                           def_is_multi_target=None, multi_target_default_val=None, qq4calc=arm_stance_qq)

            if not down_max_vertex:
                # それでも取れなければ手首位置
                return Vertex(-1, self.bones["{0}手首".format(direction)].position.copy(), MVector3D(), [], [], Bdef1(-1), -1)
        
        return down_max_vertex

    # 指先実体をはかる頂点を取得
    def get_finger_tail_vertex(self, finger_name: str, finger_tail_name: str):
        # 足首より下で、指ではないボーン
        bone_name_list = []
        direction = finger_name[0]

        if finger_name in self.bones:
            bone_name_list.append(finger_name)
        else:
            # 指ボーンがない場合、処理終了
            return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        # 腕の傾き（正確にはひじ以降の傾き）
        _, arm_stance_qq = self.calc_arm_stance("{0}手首".format(direction), finger_name)

        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_horizonal, def_is_target=None, \
                                       def_is_multi_target=None, multi_target_default_val=None, qq4calc=arm_stance_qq)
        
        if direction == "左" and right_max_vertex:
            return right_max_vertex
        
        if direction == "右" and left_max_vertex:
            return left_max_vertex

        # それでも取れなければ手首位置
        return Vertex(-1, self.bones[finger_name].position.copy(), MVector3D(), [], [], Bdef1(-1), -1)

    # ひじの厚みをはかる頂点を取得
    def get_elbow_vertex(self, direction: str):
        # 足首より下で、指ではないボーン
        bone_name_list = []

        if "{0}ひじ".format(direction) in self.bones or "{0}腕".format(direction) in self.bones:
            # 念のため、「ひじ」を含むボーンを処理対象とする
            for bk, bv in self.bones.items():
                if "{0}腕".format(direction) == bk:
                    bone_name_list.append(bk)
                if "{0}ひじ".format(direction) == bk:
                    bone_name_list.append(bk)
        else:
            # ひじボーンがない場合、処理終了
            return Vertex(-1, MVector3D(), MVector3D(), [], [], Bdef1(-1), -1)
        
        # 腕の傾き（正確にはひじ以降の傾き）
        _, arm_stance_qq = self.calc_arm_stance("{0}腕".format(direction), "{0}ひじ".format(direction))

        up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
            = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_horizonal, def_is_target=self.def_is_target_x_limit, \
                                       def_is_multi_target=self.def_is_multi_target_down_front, multi_target_default_val=MVector3D(0, 99999, 99999), qq4calc=arm_stance_qq)

        if not down_max_vertex:
            # 腕もひじが取れなかった場合、X制限なしに取得する

            up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
                back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex \
                = self.get_bone_end_vertex(bone_name_list, self.def_calc_vertex_pos_horizonal, def_is_target=None, \
                                           def_is_multi_target=None, multi_target_default_val=None, qq4calc=arm_stance_qq)

            if not down_max_vertex:
                # それでも取れなければひじ位置
                return Vertex(-1, self.bones["{0}ひじ".format(direction)].position.copy(), MVector3D(), [], [], Bdef1(-1), -1)
        
        return down_max_vertex

    # 頂点位置を返す（オリジナルそのまま）
    def def_calc_vertex_pos_original(self, b: Bone, v: Vertex, qq4calc: MQuaternion):
        return v.position

    # 水平にした場合の頂点位置を返す
    def def_calc_vertex_pos_horizonal(self, b: Bone, v: Vertex, qq4calc: MQuaternion):
        horzinal_v_pos = qq4calc.inverted() * (v.position - self.bones["{0}ひじ".format(b.name[0])].position)
        return horzinal_v_pos

    # X軸方向の制限がかかった頂点のみを対象とする
    def def_is_target_x_limit(self, b: Bone, v: Vertex, v_pos: MVector3D):
        return v_pos.x() - 0.1 <= b.position.x() <= v_pos.x() + 0.1

    # 最も底面でかつ前面にある頂点であるか
    def def_is_multi_target_down_front(self, multi_max_pos: MVector3D, v_pos: MVector3D):
        return v_pos.y() <= multi_max_pos.y() + 0.1 and v_pos.z() <= multi_max_pos.z()
    
    # 最も底面でかつ前面にある頂点であるか
    def def_is_multi_target_down_front_sole(self, multi_max_pos: MVector3D, v_pos: MVector3D):
        return v_pos.y() <= multi_max_pos.y() + 0.1 and v_pos.z() <= multi_max_pos.z()
    
    # 指定ボーンにウェイトが乗っている頂点とそのINDEX
    def get_bone_end_vertex(self, bone_name_list, def_calc_vertex_pos, def_is_target=None, def_is_multi_target=None, multi_target_default_val=None, qq4calc=None):
        # 指定ボーンにウェイトが乗っているボーンINDEXリスト
        bone_idx_list = []
        for bk, bv in self.bones.items():
            if bk in bone_name_list and bv.index in self.vertices:
                bone_idx_list.append(bv.index)

        if len(bone_idx_list) == 0:
            logger.test("bone_name: %s, ウェイト頂点がない", bone_name_list)
            # ウェイトボーンがない場合、初期値
            return MVector3D(), None, MVector3D(), None, MVector3D(), None, MVector3D(), None, MVector3D(), None, MVector3D(), None, MVector3D(), None

        logger.test("model: %s, bone_name: %s, bone_idx_list:%s", self.name, bone_name_list, bone_idx_list)

        up_max_pos = MVector3D(0, -99999, 0)
        up_max_vertex = None
        down_max_pos = MVector3D(0, 99999, 0)
        down_max_vertex = None
        right_max_pos = MVector3D(99999, 0, 0)
        right_max_vertex = None
        left_max_pos = MVector3D(-99999, 0, 0)
        left_max_vertex = None
        back_max_pos = MVector3D(0, 0, -99999)
        back_max_vertex = None
        front_max_pos = MVector3D(0, 0, 99999)
        front_max_vertex = None
        multi_max_pos = multi_target_default_val
        multi_max_vertex = None

        for bone_idx in bone_idx_list:
            if bone_idx not in self.bone_indexes:
                continue

            # ボーンINDEXに該当するボーン
            bone = self.bones[self.bone_indexes[bone_idx]]

            for v in self.vertices[bone_idx]:
                v_pos = def_calc_vertex_pos(bone, v, qq4calc)

                if def_is_target and def_is_target(bone, v, v_pos) or not def_is_target:
                    # 処理対象頂点である場合のみ判定処理に入る
                    if v_pos.y() < down_max_pos.y():
                        # 指定ボーンにウェイトが乗っていて、かつ最下の頂点より下の場合、保持
                        down_max_pos = v_pos
                        down_max_vertex = v

                    if v_pos.y() > up_max_pos.y():
                        # 指定ボーンにウェイトが乗っていて、かつ最上の頂点より上の場合、保持
                        up_max_pos = v_pos
                        up_max_vertex = v

                    if v_pos.x() < right_max_pos.x():
                        # 指定ボーンにウェイトが乗っていて、かつ最下の頂点より下の場合、保持
                        right_max_pos = v_pos
                        right_max_vertex = v

                    if v_pos.x() > left_max_pos.x():
                        # 指定ボーンにウェイトが乗っていて、かつ最上の頂点より上の場合、保持
                        left_max_pos = v_pos
                        left_max_vertex = v

                    if v_pos.z() < front_max_pos.z():
                        # 指定ボーンにウェイトが乗っていて、かつ最上の頂点より手前の場合、保持
                        front_max_pos = v_pos
                        front_max_vertex = v

                    if v_pos.z() > back_max_pos.z():
                        # 指定ボーンにウェイトが乗っていて、かつ最下の頂点より奥の場合、保持
                        back_max_pos = v_pos
                        back_max_vertex = v
                    
                    if def_is_multi_target and def_is_multi_target(multi_max_pos, v_pos):
                        multi_max_pos = v_pos
                        multi_max_vertex = v

        return up_max_pos, up_max_vertex, down_max_pos, down_max_vertex, right_max_pos, right_max_vertex, left_max_pos, left_max_vertex, \
            back_max_pos, back_max_vertex, front_max_pos, front_max_vertex, multi_max_pos, multi_max_vertex

    @classmethod
    def get_effective_value(cls, v):
        if math.isnan(v):
            return 0
        
        if math.isinf(v):
            return 0
        
        return v

    @classmethod
    def set_effective_value_vec3(cls, vec3):
        vec3.setX(cls.get_effective_value(vec3.x()))
        vec3.setY(cls.get_effective_value(vec3.y()))
        vec3.setZ(cls.get_effective_value(vec3.z()))

