# -*- coding: utf-8 -*-
#
import _pickle as cPickle
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
    
    def get_idx_list(self, weight=0):
        return [self.index0]
        
    def __str__(self):
        return "<Bdef1 {0}>".format(self.index0)


class Bdef2(Deform):
    def __init__(self, index0, index1, weight0):
        self.index0 = index0
        self.index1 = index1
        self.weight0 = weight0
        
    def get_idx_list(self, weight=0):
        idx_list = []
        if self.weight0 >= weight:
            idx_list.append(self.index0)
        if (1 - self.weight0) >= weight:
            idx_list.append(self.index1)
        return idx_list
        
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
        
    def get_idx_list(self, weight=0):
        weight_idxs = np.where(np.array([self.weight0, self.weight1, self.weight2, self.weight3]) >= weight)
        return (np.array([self.index0, self.index1, self.index2, self.index3])[weight_idxs]).tolist()

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
        
    def get_idx_list(self, weight=0):
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
        
    def get_idx_list(self, weight=0):
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
            

# 表示枠構造-----------------------
class DisplaySlot:
    def __init__(self, name, english_name, special_flag, display_type=0, references=None):
        self.name = name
        self.english_name = english_name
        self.special_flag = special_flag
        self.display_type = display_type
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
        if not bone.getConnectionFlag() and bone.tail_position > MVector3D():
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        elif bone.getConnectionFlag() and bone.tail_index >= 0 and bone.tail_index in self.bone_indexes:
            # 表示先が指定されているの場合、保持
            to_pos = self.bones[self.bone_indexes[bone.tail_index]].position
        else:
            # 表示先がない場合、とりあえず親ボーンからの向きにする
            from_pos = self.bones[self.bone_indexes[bone.parent_index]].position
            to_pos = self.bones[bone.name].position
            # for b in self.bones.values():
            #     if b.parent_index == bone.index and self.bones[self.bone_indexes[b.index]].position != bone.position:
            #         to_pos = self.bones[self.bone_indexes[b.index]].position
            #         break
        
        logger.test("get_local_x_axis: from: %s, to: %s, tail:%s-[%s]", from_pos, to_pos, bone.tail_index, bone.tail_position)
        # 軸制限の指定が無い場合、子の方向
        x_axis = (to_pos - from_pos).normalized()

        if fixed_x_axis != MVector3D() and np.sign(fixed_x_axis.x()) != np.sign(x_axis.x()):
            # 軸制限の軸方向と計算上の軸方向が違う場合、逆ベクトル
            x_axis = -fixed_x_axis

        return x_axis

    def get_local_x_qq(self, bone_name: str, base_local_axis=None):
        local_axis_qq = MQuaternion()

        # 自身から親を引いた軸の向き
        local_axis = self.get_local_x_axis(bone_name)
        if not base_local_axis:
            # 最も長い軸の向き(基本姿勢)を未指定の場合は求める
            base_local_axis = MVector3D(np.where(np.abs(local_axis.data()) == np.max(np.abs(local_axis.data())), 1 * np.sign(local_axis.data()), 0))
        local_axis_qq = MQuaternion.rotationTo(base_local_axis, local_axis.normalized())

        logger.test("get_local_x_qq: local_axis[%s], base_local_axis[%s], local_axis_qq[%s]", local_axis.to_log(), base_local_axis.to_log(), local_axis_qq.toEulerAngles4MMD().to_log())

        return local_axis_qq

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
        
        if target_bone_name not in self.bones:
            # 開始ボーン名がなければ終了
            return links

        start_type_bone = target_bone_name
        if target_bone_name.startswith("右") or target_bone_name.startswith("左"):
            # 左右から始まってたらそれは除く
            start_type_bone = target_bone_name[1:]

        # 自分をリンクに登録
        links.append(self.bones[target_bone_name].copy())

        parent_name = None
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

