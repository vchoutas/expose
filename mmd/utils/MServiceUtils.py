# -*- coding: utf-8 -*-
#
import numpy as np # noqa
import math # noqa
import numpy as np
import re
from math import sin, cos, acos, atan2, asin, pi, sqrt

from mmd.module.MParams import BoneLinks # noqa
from mmd.module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4 # noqa
from mmd.mmd.PmxData import PmxModel, Bone, Vertex, Material, Morph, DisplaySlot, RigidBody, Joint # noqa
from mmd.mmd.VmdData import VmdMotion, VmdBoneFrame, VmdCameraFrame, VmdInfoIk, VmdLightFrame, VmdMorphFrame, VmdShadowFrame, VmdShowIkFrame # noqa
from mmd.utils import MBezierUtils # noqa
from mmd.utils.MLogger import MLogger # noqa

logger = MLogger(__name__)


# IK計算
# target_pos: IKリンクの目的位置
# ik_links: IKリンク
def calc_IK(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, target_pos: MVector3D, ik_links: BoneLinks, max_count=10):
    bone_name_list = list(ik_links.all().keys())[1:]

    for bone_name in bone_name_list:
        # bfをモーションに登録
        bf = motion.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
        motion.regist_bf(bf, bone_name, fno)
    
    local_effector_pos = MVector3D()
    local_target_pos = MVector3D()

    for cnt in range(max_count):
        # 規定回数ループ
        for ik_idx, joint_name in enumerate(bone_name_list):
            # 処理対象IKボーン
            ik_bone = ik_links.get(joint_name)

            # 現在のボーングローバル位置と行列を取得
            global_3ds_dic, total_mats = calc_global_pos(model, links, motion, fno, limit_links=None, return_matrix=True, is_local_x=False)

            # エフェクタ（末端）
            global_effector_pos = global_3ds_dic[ik_links.first_name()]

            # 注目ノード（実際に動かすボーン）
            joint_mat = total_mats[joint_name]

            # ワールド座標系から注目ノードの局所座標系への変換
            inv_coord = joint_mat.inverted()

            # 注目ノードを起点とした、エフェクタのローカル位置
            local_effector_pos = inv_coord * global_effector_pos
            local_target_pos = inv_coord * target_pos

            #  (1) 基準関節→エフェクタ位置への方向ベクトル
            basis2_effector = local_effector_pos.normalized()
            #  (2) 基準関節→目標位置への方向ベクトル
            basis2_target = local_target_pos.normalized()

            # ベクトル (1) を (2) に一致させるための最短回転量（Axis-Angle）
            # 回転角
            rotation_dot = MVector3D.dotProduct(basis2_effector, basis2_target)
            # 回転角度
            rotation_radian = acos(max(-1, min(1, rotation_dot)))

            if abs(rotation_radian) > 0.0001:
                # 一定角度以上の場合

                # 回転軸
                rotation_axis = MVector3D.crossProduct(basis2_effector, basis2_target).normalized()
                # 回転角度
                rotation_degree = math.degrees(rotation_radian)

                # 関節回転量の補正(最大変位量を制限する)
                correct_qq = MQuaternion.fromAxisAndAngle(rotation_axis, min(rotation_degree, ik_bone.degree_limit))

                # ジョイントに補正をかける
                bf = motion.calc_bf(joint_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
                new_ik_qq = correct_qq * bf.rotation

                # IK軸制限がある場合、上限下限をチェック
                if ik_bone.ik_limit_min != MVector3D() and ik_bone.ik_limit_max != MVector3D():
                    x_qq, y_qq, z_qq, yz_qq = separate_local_qq(fno, ik_bone.name, new_ik_qq, model.get_local_x_axis(ik_bone.name))

                    logger.test("new_ik_qq: %s, x_qq: %s, y_qq: %s, z_qq: %s", new_ik_qq.toEulerAngles(), x_qq.toEulerAngles(), y_qq.toEulerAngles(), z_qq.toEulerAngles())
                    logger.test("new_ik_qq: %s, x_qq: %s, y_qq: %s, z_qq: %s", new_ik_qq.toDegree(), x_qq.toDegree(), y_qq.toDegree(), z_qq.toDegree())

                    euler_x = min(ik_bone.ik_limit_max.x(), max(ik_bone.ik_limit_min.x(), x_qq.toDegree()))
                    euler_y = min(ik_bone.ik_limit_max.y(), max(ik_bone.ik_limit_min.y(), y_qq.toDegree()))
                    euler_z = min(ik_bone.ik_limit_max.z(), max(ik_bone.ik_limit_min.z(), z_qq.toDegree()))

                    logger.test("limit_qq: %s -> %s", new_ik_qq.toEulerAngles(), MQuaternion.fromEulerAngles(euler_x, euler_y, euler_z).toEulerAngles())

                    new_ik_qq = MQuaternion.fromEulerAngles(euler_x, euler_y, euler_z)

                bf.rotation = new_ik_qq

        # 位置の差がほとんどない場合、終了
        if (local_effector_pos - local_target_pos).lengthSquared() < 0.0001:
            return
        
    return


# クォータニオンをローカル軸の回転量に分離
def separate_local_qq(fno: int, bone_name: str, qq: MQuaternion, global_x_axis: MVector3D):
    # ローカル座標系（ボーンベクトルが（1，0，0）になる空間）の向き
    local_axis = MVector3D(1, 0, 0)
    
    # グローバル座標系（Ａスタンス）からローカル座標系（ボーンベクトルが（1，0，0）になる空間）への変換
    global2local_qq = MQuaternion.rotationTo(global_x_axis, local_axis)
    local2global_qq = MQuaternion.rotationTo(local_axis, global_x_axis)

    # X成分を抽出する ------------

    mat_x1 = MMatrix4x4()
    mat_x1.setToIdentity()              # 初期化
    mat_x1.rotate(qq)                   # 入力qq
    mat_x1.translate(global_x_axis)     # グローバル軸方向に伸ばす
    mat_x1_vec = mat_x1 * MVector3D()

    # YZの回転量（自身のねじれを無視する）
    yz_qq = MQuaternion.rotationTo(global_x_axis, mat_x1_vec)

    # 除去されたX成分を求める
    mat_x2 = MMatrix4x4()
    mat_x2.setToIdentity()              # 初期化
    mat_x2.rotate(qq)                   # 元々の回転量

    mat_x3 = MMatrix4x4()
    mat_x3.setToIdentity()              # 初期化
    mat_x3.rotate(yz_qq)                # YZの回転量

    x_qq = (mat_x2 * mat_x3.inverted()).toQuaternion()

    # YZ回転からZ成分を抽出する --------------

    mat_z1 = MMatrix4x4()
    mat_z1.setToIdentity()              # 初期化
    mat_z1.rotate(yz_qq)                # YZの回転量
    mat_z1.rotate(global2local_qq)      # グローバル軸の回転量からローカルの回転量に変換
    mat_z1.translate(local_axis)        # ローカル軸方向に伸ばす
    
    mat_z1_vec = mat_z1 * MVector3D()
    mat_z1_vec.setZ(0)                  # Z方向の移動量を潰す

    # ローカル軸からZを潰した移動への回転量
    local_z_qq = MQuaternion.rotationTo(local_axis, mat_z1_vec)

    # ボーンローカル座標系の回転をグローバル座標系の回転に戻す
    mat_z2 = MMatrix4x4()
    mat_z2.setToIdentity()              # 初期化
    mat_z2.rotate(local_z_qq)           # ローカル軸上のZ回転
    mat_z2.rotate(local2global_qq)      # ローカル軸上からグローバル軸上に変換

    z_qq = mat_z2.toQuaternion()

    # YZ回転からY成分だけ取り出す -----------
    
    mat_y1 = MMatrix4x4()
    mat_y1.setToIdentity()              # 初期化
    mat_y1.rotate(yz_qq)                # グローバルYZの回転量

    mat_y2 = MMatrix4x4()
    mat_y2.setToIdentity()              # 初期化
    mat_y2.rotate(z_qq)                 # グローバルZの回転量
    mat_y2_qq = (mat_y1 * mat_y2.inverted()).toQuaternion()

    # X成分の捻れが混入したので、XY回転からYZ回転を取り出すことでXキャンセルをかける。
    mat_y3 = MMatrix4x4()
    mat_y3.setToIdentity()
    mat_y3.rotate(mat_y2_qq)
    mat_y3.translate(global_x_axis)
    mat_y3_vec = mat_y3 * MVector3D()

    y_qq = MQuaternion.rotationTo(global_x_axis, mat_y3_vec)

    # Xを再度求める -------------

    mat_x4 = MMatrix4x4()
    mat_x4.setToIdentity()
    mat_x4.rotate(qq)

    mat_x5 = MMatrix4x4()
    mat_x5.setToIdentity()
    mat_x5.rotate(y_qq)

    mat_x6 = MMatrix4x4()
    mat_x6.setToIdentity()
    mat_x6.rotate(z_qq)

    x_qq = (mat_x5.inverted() * mat_x4 * mat_x6.inverted()).toQuaternion()

    return x_qq, y_qq, z_qq, yz_qq

# 正面向きの情報を含むグローバル位置
def calc_front_global_pos(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, limit_links=None, direction_limit_links=None):
    # グローバル位置
    (global_3ds, total_mats) = calc_global_pos(model, links, motion, fno, limit_links, False, False)
    org_center_global_3ds = global_3ds

    # 指定ボーンまでの向いている回転量（回転のみの制限がかかっている場合、それを優先）
    direction_qq = calc_direction_qq(model, links, motion, fno, (limit_links if not direction_limit_links else direction_limit_links))

    # 正面向きのグローバル位置
    front_global_3ds = calc_global_pos_by_direction(direction_qq.inverted(), org_center_global_3ds)

    return global_3ds, front_global_3ds, direction_qq


# グローバル位置算出
def calc_global_pos(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, limit_links=None, return_matrix=False, is_local_x=False):
    # pfun = profile(c_calc_relative_position)
    # cdef list trans_vs = pfun(model, links, motion, fno, limit_links)
    trans_vs = calc_relative_position(model, links, motion, fno, limit_links)
    add_qs = calc_relative_rotation(model, links, motion, fno, limit_links)

    # 行列
    matrixs = [MMatrix4x4() for i in range(links.size())]

    for n, (lname, v, q) in enumerate(zip(links.all().keys(), trans_vs, add_qs)):
        # 行列を生成
        mm = MMatrix4x4()
        # 初期化
        mm.setToIdentity()
        # 移動
        mm.translate(v)
        # 回転
        mm.rotate(q)
        # 設定
        matrixs[n] = mm

    total_mats = {}
    global_3ds_dic = {}
    mm = MMatrix4x4()

    for n, (lname, v) in enumerate(zip(links.all().keys(), trans_vs)):
        if n == 0:
            mm = MMatrix4x4()
            mm.setToIdentity()

        for m in range(n):
            # 最後のひとつ手前までループ
            if m == 0:
                # 0番目の位置を初期値とする
                mm = matrixs[0].copy()
            else:
                # 自分より前の行列結果を掛け算する
                mm *= matrixs[m]
        
        # 自分は、位置だけ掛ける
        global_3ds_dic[lname] = mm * v

        # 最後の行列をかけ算する
        total_mats[lname] = mm * matrixs[n]

        # ローカル軸の向きを調整する
        if n > 0 and is_local_x:
            # ボーン自身にローカル軸が設定されているか
            local_x_matrix = MMatrix4x4()
            local_x_matrix.setToIdentity()

            local_axis_qq = MQuaternion()

            if model.bones[lname].local_x_vector == MVector3D():
                # ローカル軸が設定されていない場合、計算

                # 自身から親を引いた軸の向き
                local_axis = model.bones[lname].position - links.get(lname, offset=-1).position
                local_axis_qq = MQuaternion.fromDirection(local_axis.normalized(), MVector3D(0, 0, 1))
            else:
                # ローカル軸が設定されている場合、その値を採用
                local_axis_qq = MQuaternion.fromDirection(model.bones[lname].local_x_vector.normalized(), MVector3D(0, 0, 1))
            
            local_x_matrix.rotate(local_axis_qq)

            total_mats[lname] *= local_x_matrix

    if not return_matrix:
        return global_3ds_dic

    return global_3ds_dic, total_mats


# 指定された方向に向いた場合の位置情報を返す
def calc_global_pos_by_direction(direction_qq: MQuaternion, target_pos_3ds_dic: dict):
    direction_pos_dic = {}

    for bone_name, target_pos in target_pos_3ds_dic.items():
        # # その地点の回転後の位置
        # direction_pos_dic[bone_name] = direction_qq * target_pos
        mat = MMatrix4x4()
        # 初期化
        mat.setToIdentity()
        # 指定位置
        mat.translate(target_pos)
        # 回転させる
        mat.rotate(direction_qq)
        # その地点の回転後の位置
        direction_pos_dic[bone_name] = mat * MVector3D()
        # logger.test("f: %s, direction_qq: %s", bone_name, direction_qq.toEulerAngles4MMD())
        # logger.test("f: %s, target_pos: %s", bone_name, target_pos)
        # logger.test("f: %s, direction_pos_dic: %s", bone_name, direction_pos_dic[bone_name])
    
    return direction_pos_dic


# 各ボーンの相対位置情報
def calc_relative_position(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, limit_links=None):
    trans_vs = []

    for link_idx, link_bone_name in enumerate(links.all()):
        link_bone = links.get(link_bone_name)

        if not limit_links or (limit_links and limit_links.get(link_bone_name)):
            # 上限リンクがある倍、ボーンが存在している場合のみ、モーション内のキー情報を取得
            fill_bf = motion.calc_bf(link_bone.name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
        else:
            # 上限リンクでボーンがない場合、ボーンは初期値
            fill_bf = VmdBoneFrame(fno=fno)
            fill_bf.set_name(link_bone_name)

        # 位置
        if link_idx == 0:
            # 一番親は、グローバル座標を考慮
            trans_vs.append(link_bone.position + fill_bf.position)
        else:
            # 位置：自身から親の位置を引いた相対位置
            trans_vs.append(link_bone.position + fill_bf.position - links.get(link_bone_name, offset=-1).position)

    return trans_vs


# 各ボーンの相対回転情報
def calc_relative_rotation(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, limit_links=None):
    add_qs = []

    for link_idx, link_bone_name in enumerate(links.all()):
        link_bone = links.get(link_bone_name)

        if not limit_links or (limit_links and limit_links.get(link_bone_name)):
            # 上限リンクがある場合、ボーンが存在している場合のみ、モーション内のキー情報を取得
            fill_bf = motion.calc_bf(link_bone.name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
        else:
            # 上限リンクでボーンがない場合、ボーンは初期値
            fill_bf = VmdBoneFrame(fno=fno)
            fill_bf.set_name(link_bone_name)
        
        # 実際の回転量を計算
        rot = deform_rotation(model, motion, fill_bf)

        add_qs.append(rot)

    return add_qs


# 指定ボーンの実際の回転情報
def deform_rotation(model: PmxModel, motion: VmdMotion, bf: VmdBoneFrame):
    if bf.name not in model.bones:
        return MQuaternion()

    bone = model.bones[bf.name]
    rot = bf.rotation.normalized().copy()

    rot = deform_fix_rotation(bf.name, bone.fixed_axis, rot)

    if bone.getExternalRotationFlag() and bone.effect_index in model.bone_indexes:
        
        effect_parent_bone = bone
        effect_bone = model.bones[model.bone_indexes[bone.effect_index]]
        cnt = 0

        while cnt < 100:
            # 付与親が取得できたら、該当する付与親の回転を取得する
            effect_bf = motion.calc_bf(effect_bone.name, bf.fno, is_key=False, is_read=False, is_reset_interpolation=False)

            # 自身の回転量に付与親の回転量を付与率を加味して付与する
            if effect_parent_bone.effect_factor == 0:
                # ゼロの場合、とりあえず初期化
                logger.debug(f"モデル「{model.name}」ボーン「{effect_parent_bone.name}」の付与率がゼロ")
                rot = MQuaternion()
            elif effect_parent_bone.effect_factor < 0:
                # マイナス付与の場合、逆回転
                rot = rot * (effect_bf.rotation * abs(effect_parent_bone.effect_factor)).inverted()
            else:
                rot = rot * (effect_bf.rotation * effect_parent_bone.effect_factor)

            if effect_bone.getExternalRotationFlag() and effect_bone.effect_index in model.bone_indexes:
                # 付与親の親として現在のeffectboneを保持
                effect_parent_bone = effect_bone
                # 付与親置き換え
                effect_bone = model.bones[model.bone_indexes[effect_bone.effect_index]]
            else:
                break

            cnt += 1

    return rot


# 軸制限回転を求め直す
def deform_fix_rotation(bone_name: str, fixed_axis: MVector3D, rot: MQuaternion):
    if fixed_axis != MVector3D():
        # 回転角度を求める
        if rot != MQuaternion():
            # 回転補正
            if "右" in bone_name and rot.x() > 0 and fixed_axis.x() <= 0:
                rot.setX(rot.x() * -1)
                rot.setScalar(rot.scalar() * -1)
            elif "左" in bone_name and rot.x() < 0 and fixed_axis.x() >= 0:
                rot.setX(rot.x() * -1)
                rot.setScalar(rot.scalar() * -1)
            # 回転補正（コロン式ミクさん等軸反転パターン）
            elif "右" in bone_name and rot.x() < 0 and fixed_axis.x() > 0:
                rot.setX(rot.x() * -1)
                rot.setScalar(rot.scalar() * -1)
            elif "左" in bone_name and rot.x() > 0 and fixed_axis.x() < 0:
                rot.setX(rot.x() * -1)
                rot.setScalar(rot.scalar() * -1)
            
            rot.normalize()
        
        # 軸固定の場合、回転を制限する
        rot = MQuaternion.fromAxisAndAngle(fixed_axis, rot.toDegree())

    return rot


# 指定されたボーンまでの回転量
def calc_direction_qq(model: PmxModel, links: BoneLinks, motion: VmdMotion, fno: int, limit_links=None):
    add_qs = calc_relative_rotation(model, links, motion, fno, limit_links)

    total_qq = MQuaternion()
    for qq in add_qs:
        total_qq *= qq

    return total_qq.normalized()


# ファイルのエンコードを取得する
def get_file_encoding(file_path):

    try: 
        f = open(file_path, "rb")
        fbytes = f.read()
        f.close()
    except:
        raise Exception("unknown encoding!")
        
    codelst = ('utf_8', 'shift-jis')
    
    for encoding in codelst:
        try:
            fstr = fbytes.decode(encoding) # bytes文字列から指定文字コードの文字列に変換
            fstr = fstr.encode('utf-8') # uft-8文字列に変換
            # 問題なく変換できたらエンコードを返す
            logger.debug("%s: encoding: %s", file_path, encoding)
            return encoding
        except Exception as e:
            print(e)
            pass
            
    raise Exception("unknown encoding!")


# 数値でソート
def sort_by_numeric(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


