# -*- coding: utf-8 -*-
#
import math
import numpy as np
import struct
import _pickle as cPickle

from mmd.utils import MBezierUtils # noqa
from mmd.utils.MLogger import MLogger

from mmd.module.MMath import MRect, MVector2D, MVector3D, MVector4D, MQuaternion, MMatrix4x4, get_effective_value # noqa

logger = MLogger(__name__, level=1)

# OneEuroFilter
# オリジナル：https://www.cristal.univ-lille.fr/~casiez/1euro/
# ----------------------------------------------------------------------------

class LowPassFilter:

    def __init__(self, alpha):
        self.__setAlpha(alpha)
        self.__y = -1
        self.__s = -1

    def __setAlpha(self, alpha):
        alpha = max(0.000001, min(1, alpha))
        if alpha <= 0 or alpha > 1.0:
            raise ValueError("alpha (%s) should be in (0.0, 1.0]" % alpha)
        self.__alpha = alpha

    def __call__(self, value: float, timestamp=-1, alpha=-1):
        s = 0
        if alpha >= 0:
            self.__setAlpha(alpha)
        if self.__y < 0:
            s = value
        else:
            s = self.__alpha * value + (1.0 - self.__alpha) * self.__s
        self.__y = value
        self.__s = s
        return s

    def lastValue(self):
        return self.__y

    # IK用処理スキップ
    def skip(self, value: float, timestamp=-1, alpha=-1):
        self.__y = value
        self.__s = value

        return value


class OneEuroFilter:

    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        if freq <= 0:
            raise ValueError("freq should be >0")
        if mincutoff <= 0:
            raise ValueError("mincutoff should be >0")
        if dcutoff <= 0:
            raise ValueError("dcutoff should be >0")
        self.__freq = freq
        self.__mincutoff = mincutoff
        self.__beta = beta
        self.__dcutoff = dcutoff
        self.__x = LowPassFilter(self.__alpha(self.__mincutoff))
        self.__dx = LowPassFilter(self.__alpha(self.__dcutoff))
        self.__lasttime = -1

    def __alpha(self, cutoff: float):
        te = 1.0 / self.__freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, timestamp=-1):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        # ---- estimate the current variation per second
        prev_x = self.__x.lastValue()
        dx = 0.0 if prev_x < 0 else (x - prev_x) * self.__freq  # FIXME: 0.0 or value?
        edx = self.__dx(dx, timestamp, alpha=self.__alpha(self.__dcutoff))
        # ---- use it to update the cutoff frequency
        cutoff = self.__mincutoff + self.__beta * math.fabs(edx)
        # まったく同じ値の場合、スキップ
        if prev_x == x:
            return self.__x.skip(x, timestamp, alpha=self.__alpha(cutoff))
        # ---- filter the given value
        return self.__x(x, timestamp, alpha=self.__alpha(cutoff))

    # IK用処理スキップ
    def skip(self, x: float, timestamp=-1):
        # ---- update the sampling frequency based on timestamps
        if self.__lasttime and timestamp and self.__lasttime != timestamp:
            self.__freq = 1.0 / (timestamp - self.__lasttime)
        self.__lasttime = timestamp
        prev_x = self.__x.lastValue()
        self.__dx.skip(prev_x)
        self.__x.skip(x)


class VmdBoneFrame:

    def __init__(self, fno=0):
        self.name = ''
        self.bname = b''
        self.fno = fno
        self.position = MVector3D()
        self.rotation = MQuaternion()
        self.org_rotation = MQuaternion()
        self.interpolation = [20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0] # noqa
        self.org_interpolation = [20, 20, 0, 0, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107, 0, 0, 0] # noqa
        # 登録対象であるか否か
        self.key = False
        # VMD読み込み処理で読み込んだキーか
        self.read = False
        # 接触回避の方向
        self.avoidance = ""
    
    def set_name(self, name):
        self.name = name
        self.bname = b'' if not name else name.encode('cp932').decode('shift_jis').encode('shift_jis')[:15].ljust(15, b'\x00')
    
    def copy(self):
        bf = VmdBoneFrame(self.fno)
        bf.name = self.name
        bf.bname = self.bname
        bf.position = self.position.copy()
        bf.rotation = self.rotation.copy()
        bf.org_rotation = self.org_rotation.copy()
        bf.interpolation = cPickle.loads(cPickle.dumps(self.interpolation, -1))
        bf.key = self.key
        bf.read = self.read

        return bf

    def __str__(self):
        return "<VmdBoneFrame name:{0}, fno:{1}, position:{2}, rotation:{3}, euler:{4}, key:{5}, read:{6}, interpolation: {7}>".format( \
            self.name, self.fno, self.position, self.rotation, self.rotation.toEulerAngles4MMD(), self.key, self.read, self.interpolation)

    def write(self, fout):
        if not self.bname:
            self.bname = self.name.encode('cp932').decode('shift_jis').encode('shift_jis')[:15].ljust(15, b'\x00')   # 15文字制限
        fout.write(self.bname)
        fout.write(struct.pack('<L', int(self.fno)))
        fout.write(struct.pack('<f', float(self.position.x())))
        fout.write(struct.pack('<f', float(self.position.y())))
        fout.write(struct.pack('<f', float(self.position.z())))
        v = self.rotation.normalized().toVector4D()
        fout.write(struct.pack('<f', float(v.x())))
        fout.write(struct.pack('<f', float(v.y())))
        fout.write(struct.pack('<f', float(v.z())))
        fout.write(struct.pack('<f', float(v.w())))
        fout.write(bytearray([int(min(127, max(0, x))) for x in self.interpolation]))


class VmdMorphFrame:
    def __init__(self, fno=0):
        self.name = ''
        self.bname = ''
        self.fno = fno
        self.ratio = 0
    
    def write(self, fout):
        if not self.bname:
            self.bname = self.name.encode('cp932').decode('shift_jis').encode('shift_jis')[:15].ljust(15, b'\x00')   # 15文字制限
        fout.write(self.bname)
        fout.write(struct.pack('<L', int(self.fno)))
        fout.write(struct.pack('<f', float(self.ratio)))

    def set_name(self, name):
        self.name = name
        self.bname = '' if not name else name.encode('cp932').decode('shift_jis').encode('shift_jis')[:15].ljust(15, b'\x00')
    
    def __str__(self):
        return "<VmdMorphFrame name:{0}, fno:{1}, ratio:{2}".format(self.name, self.fno, self.ratio)


class VmdCameraFrame:
    def __init__(self):
        self.fno = 0
        self.length = 0
        self.position = MVector3D(0, 0, 0)
        self.euler = MVector3D(0, 0, 0)
        self.interpolation = [20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107, 20, 107]
        self.angle = 0
        self.perspective = 0
        self.org_length = 0
        self.org_position = MVector3D(0, 0, 0)
        self.ratio = 0

    def write(self, fout):
        fout.write(struct.pack('<L', int(self.fno)))
        fout.write(struct.pack('<f', float(self.length)))
        fout.write(struct.pack('<f', float(self.position.x())))
        fout.write(struct.pack('<f', float(self.position.y())))
        fout.write(struct.pack('<f', float(self.position.z())))
        fout.write(struct.pack('<f', float(self.euler.x())))
        fout.write(struct.pack('<f', float(self.euler.y())))
        fout.write(struct.pack('<f', float(self.euler.z())))
        fout.write(bytearray([int(min(127, max(0, x))) for x in self.interpolation]))
        fout.write(struct.pack('<L', int(self.angle)))
        fout.write(struct.pack('b', self.perspective))

    def copy(self):
        cf = VmdCameraFrame()
        cf.fno = self.fno
        cf.length = self.length
        cf.position = self.position.copy()
        cf.euler = self.euler.copy()
        cf.interpolation = cPickle.loads(cPickle.dumps(self.interpolation, -1))
        cf.angle = self.angle
        cf.perspective = self.perspective
        cf.org_length = self.org_length
        cf.org_position = self.org_position.copy()
        cf.ratio = self.ratio

        return cf


class VmdLightFrame:
    def __init__(self):
        self.fno = 0
        self.color = MVector3D(0, 0, 0)
        self.position = MVector3D(0, 0, 0)

    def write(self, fout):
        fout.write(struct.pack('<L', self.fno))
        fout.write(struct.pack('<f', self.color.x()))
        fout.write(struct.pack('<f', self.color.y()))
        fout.write(struct.pack('<f', self.color.z()))
        fout.write(struct.pack('<f', self.position.x()))
        fout.write(struct.pack('<f', self.position.y()))
        fout.write(struct.pack('<f', self.position.z()))


class VmdShadowFrame:
    def __init__(self):
        self.fno = 0
        self.type = 0
        self.distance = 0

    def write(self, fout):
        fout.write(struct.pack('<L', self.fno))
        fout.write(struct.pack('<f', self.type))
        fout.write(struct.pack('<f', self.distance))


# VmdShowIkFrame のikの中の要素
class VmdInfoIk:
    def __init__(self, name='', onoff=0):
        self.bname = b''
        self.name = name
        self.onoff = onoff


class VmdShowIkFrame:
    def __init__(self):
        self.fno = 0
        self.show = 0
        self.ik_count = 0
        self.ik = []

    def write(self, fout):
        fout.write(struct.pack('<L', self.fno))
        fout.write(struct.pack('b', self.show))
        fout.write(struct.pack('<L', len(self.ik)))
        for k in (self.ik):
            if not k.bname:
                k.bname = k.name.encode('cp932').decode('shift_jis').encode('shift_jis')[:20].ljust(20, b'\x00')   # 20文字制限
            fout.write(k.bname)
            fout.write(struct.pack('b', k.onoff))
        

# https://blog.goo.ne.jp/torisu_tetosuki/e/bc9f1c4d597341b394bd02b64597499d
# https://w.atwiki.jp/kumiho_k/pages/15.html
class VmdMotion:
    def __init__(self):
        self.path = ''
        self.signature = ''
        self.model_name = ''
        self.last_motion_frame = 0
        self.motion_cnt = 0
        # ボーン名：VmdBoneFrameの辞書(key:ボーン名)
        self.bones = {}
        self.morph_cnt = 0
        # モーフ名：VmdMorphFrameの辞書(key:モーフ名)
        self.morphs = {}
        self.camera_cnt = 0
        # カメラ：VmdCameraFrameの配列
        self.cameras = {}
        self.light_cnt = 0
        # 照明：VmdLightFrameの配列
        self.lights = []
        self.shadow_cnt = 0
        # セルフ影：VmdShadowFrameの配列
        self.shadows = []
        self.ik_cnt = 0
        # モデル表示・IK on/off：VmdShowIkFrameの配列
        self.showiks = []
        # ハッシュ値
        self.digest = None
    
    def regist_full_bf(self, data_set_no: int, bone_name_list: list, offset=1, is_key=True):
        # 指定された全部のボーンのキーフレ取得
        fnos = self.get_bone_fnos(*bone_name_list)

        if len(fnos) == 0:
            return

        # オフセット単位でキーフレ計算
        fnos.extend(x for x in range(fnos[-1])[::offset])
        # 重複を除いて再計算
        fnos = sorted(list(set(fnos)))

        # 指定ボーン名でキーフレ登録
        for bone_name in bone_name_list:
            prev_sep_fno = 0

            for fno in fnos:
                bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
                self.regist_bf(bf, bone_name, fno, copy_interpolation=False)

                if not is_key and not bf.read:
                    # 無効化のままの場合、キーをOFFにしておく
                    self.bones[bone_name][fno].key = False

                if fno // 500 > prev_sep_fno and fnos[-1] > 0:
                    if data_set_no == 0:
                        logger.info("-- %sフレーム目:終了(%s％)【全打ち - %s】", fno, round((fno / fnos[-1]) * 100, 3), bone_name)
                        prev_sep_fno = fno // 500
                    elif data_set_no > 0:
                        logger.info("-- %sフレーム目:終了(%s％)【No.%s - 全打ち - %s】", fno, round((fno / fnos[-1]) * 100, 3), data_set_no, bone_name)
                        prev_sep_fno = fno // 500

    def get_differ_fnos(self, data_set_no: int, bone_name_list: list, limit_degrees: float, limit_length: float):
        # limit_radians = cmath.cos(math.radians(limit_degrees))
        fnos = [0]
        prev_sep_fno = 0

        # 有効キーを取得
        bone_fnos = self.get_bone_fnos(*bone_name_list, is_key=True)

        if len(bone_fnos) <= 0:
            return []
        
        # 比較対象bf
        rot_diff = 0
        mov_diff = 0
        prev_bf = None
        last_fno = bone_fnos[-1] + 1
        for fno in range(1, last_fno + 1):
            for bone_name in bone_name_list:
                prev_bf = self.calc_bf(bone_name, fno - 1, is_key=False, is_read=False, is_reset_interpolation=False)
                bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)

                if bf.read:
                    # 読み込みキーである場合、必ず処理対象に追加
                    fnos.append(fno)
                    rot_diff = 0
                    mov_diff = 0
                else:
                    # 読み込みキーではない場合、処理対象にするかチェック
                    if fno - 1 in fnos:
                        # 前のキーがある場合、とりあえずスルー
                        continue

                    # 読み込みキーとの差
                    rot_diff += abs(prev_bf.rotation.toDegree() - bf.rotation.toDegree())
                    if rot_diff > limit_degrees and limit_degrees > 0:
                        # 前と今回の内積の差が指定度数より離れている場合、追加
                        logger.debug("★ 追加 set: %s, %s, f: %s, diff: %s", data_set_no, bone_name, fno, rot_diff)
                        fnos.append(fno)
                        rot_diff = 0
                    elif limit_length > 0:
                        # 読み込みキーとの差
                        mov_diff += prev_bf.position.distanceToPoint(bf.position)
                        if mov_diff > limit_length:
                            # 前と今回の移動量の差が指定値より離れている場合、追加
                            logger.test("★ 追加 set: %s, %s, f: %s, diff: %s", data_set_no, bone_name, fno, mov_diff)
                            fnos.append(fno)
                            mov_diff = 0
                    else:
                        logger.test("× 追加なし set: %s, %s, f: %s, rot_diff: %s, mov_diff: %s", data_set_no, bone_name, fno, rot_diff, mov_diff)

                if fno // 2000 > prev_sep_fno and bone_fnos[-1] > 0:
                    if data_set_no > 0:
                        logger.info("-- %sフレーム目:終了(%s％)【No.%s - キーフレ追加準備 - %s】", fno, round((fno / bone_fnos[-1]) * 100, 3), data_set_no, bone_name)
                        prev_sep_fno = fno // 2000
                    else:
                        logger.info("-- %sフレーム目:終了(%s％)【キーフレ追加準備 - %s】", fno, round((fno / bone_fnos[-1]) * 100, 3), bone_name)
                        prev_sep_fno = fno // 2000

        # 重複を除いて再計算
        return sorted(list(set(fnos)))

    # 指定ボーンが跳ねてたりするのを回避
    def smooth_bf(self, data_set_no: int, bone_name: str, is_rot: bool, is_mov: bool, limit_degrees: float, start_fno=-1, end_fno=-1, is_show_log=True):
        # キーフレを取得する
        if start_fno < 0 and end_fno < 0:
            # 範囲指定がない場合、全範囲
            fnos = self.get_bone_fnos(bone_name)
        else:
            # 範囲指定がある場合はその範囲内だけ
            fnos = self.get_bone_fnos(bone_name, start_fno=start_fno, end_fno=end_fno)
        
        limit_radians = math.radians(limit_degrees)

        prev_sep_fno = 0

        if len(fnos) > 2:
            for fno in fnos:
                prev_bf = self.calc_bf(bone_name, fno - 1, is_key=False, is_read=False, is_reset_interpolation=False)
                now_bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
                next_bf = self.calc_bf(bone_name, fno + 1, is_key=False, is_read=False, is_reset_interpolation=False)

                if is_rot and now_bf.key:
                    # 前後の内積
                    prev_next_dot = MQuaternion.dotProduct(prev_bf.rotation, next_bf.rotation)
                    # 自分と後の内積
                    now_next_dot = MQuaternion.dotProduct(now_bf.rotation, next_bf.rotation)
                    # 内積差分
                    diff = np.abs(np.diff([prev_next_dot, now_next_dot]))
                    logger.test("set: %s, %s, f: %s, diff: %s, prev_next_dot: %s, now_next_dot: %s", data_set_no, bone_name, fno, diff, prev_next_dot, now_next_dot)

                    # 前後と自分の内積の差が一定以上の場合、円滑化
                    if prev_next_dot > now_next_dot and diff > limit_radians:
                        logger.debug("★ 円滑化 set: %s, %s, f: %s, diff: %s, prev_next_dot: %s, now_next_dot: %s", \
                                     data_set_no, bone_name, fno, diff, prev_next_dot, now_next_dot)

                        now_bf.rotation = MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ((now_bf.fno - prev_bf.fno) / (next_bf.fno - prev_bf.fno)))
                
                if is_show_log and data_set_no > 0 and fno // 2000 > prev_sep_fno and fnos[-1] > 0:
                    logger.info("-- %sフレーム目:終了(%s％)【No.%s - 円滑化 - %s】", fno, round((fno / fnos[-1]) * 100, 3), data_set_no, bone_name)
                    prev_sep_fno = fno // 2000

    # フィルターをかける
    def smooth_filter_bf(self, data_set_no: int, bone_name: str, is_rot: bool, is_mov: bool, loop=1, \
                         config={"freq": 30, "mincutoff": 0.3, "beta": 0.01, "dcutoff": 0.25}, start_fno=-1, end_fno=-1, is_show_log=True):
        prev_sep_fno = 0

        for n in range(loop):
            # 移動用フィルタ
            pxfilter = OneEuroFilter(**config)
            pyfilter = OneEuroFilter(**config)
            pzfilter = OneEuroFilter(**config)

            # 回転用フィルタ
            rxfilter = OneEuroFilter(**config)
            ryfilter = OneEuroFilter(**config)
            rzfilter = OneEuroFilter(**config)

            fnos = self.get_bone_fnos(bone_name)
            prev_sep_fno = 0

            # キーフレを取得する
            if start_fno < 0 and end_fno < 0:
                # 範囲指定がない場合、全範囲
                fnos = self.get_bone_fnos(bone_name)
            else:
                # 範囲指定がある場合はその範囲内だけ
                fnos = self.get_bone_fnos(bone_name, start_fno=start_fno, end_fno=end_fno)

            # 全区間をフィルタにかける
            for fno in fnos:
                now_bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)

                if is_mov:
                    # 移動XYZそれぞれにフィルターをかける
                    px = pxfilter(now_bf.position.x(), fno)
                    py = pyfilter(now_bf.position.y(), fno)
                    pz = pzfilter(now_bf.position.z(), fno)
                    now_bf.position = MVector3D(px, py, pz)
                
                if is_rot:
                    # 回転XYZそれぞれにフィルターをかける(オイラー角)
                    r = now_bf.rotation.toEulerAngles()
                    rx = rxfilter(r.x(), fno)
                    ry = ryfilter(r.y(), fno)
                    rz = rzfilter(r.z(), fno)

                    # クォータニオンに戻して保持
                    new_qq = MQuaternion.fromEulerAngles(rx, ry, rz)
                    now_bf.rotation = new_qq

                if is_show_log and data_set_no > 0 and fno // 2000 > prev_sep_fno and fnos[-1] > 0:
                    logger.info("-- %sフレーム目:終了(%s％)【No.%s - フィルタリング - %s(%s)】", fno, round((fno / fnos[-1]) * 100, 3), data_set_no, bone_name, (n + 1))
                    prev_sep_fno = fno // 2000

    # 無効なキーを物理削除する
    def remove_unkey_bf(self, data_set_no: int, bone_name: str):
        for fno in self.get_bone_fnos(bone_name):
            bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)

            if fno in self.bones[bone_name] and not bf.key:
                del self.bones[bone_name][fno]

    # 指定ボーンの不要キーを削除する
    # 変曲点を求める
    # https://teratail.com/questions/162391
    def remove_unnecessary_bf(self, data_set_no: int, bone_name: str, is_rot: bool, is_mov: bool, \
                              offset=0, rot_diff_limit=0.1, mov_diff_limit=0.1, r_start_fno=-1, r_end_fno=-1, is_show_log=True, is_force=False):
        prev_sep_fno = 0

        # キーフレを取得する
        if r_start_fno < 0 and r_end_fno < 0:
            # 範囲指定がない場合、全範囲
            fnos = self.get_bone_fnos(bone_name)
        else:
            # 範囲指定がある場合はその範囲内だけ
            fnos = self.get_bone_fnos(bone_name, start_fno=r_start_fno, end_fno=r_end_fno)
        logger.debug("remove_unnecessary_bf fnos: %s, %s", bone_name, fnos)
        
        if len(fnos) <= 1:
            return
        
        bf = None
        prev_bf = None

        fno = fnos[0] + 1
        start_fno = fnos[0]
        last_fno = fnos[0]
        rot_values = [0]
        mx_values = [0]
        my_values = [0]
        mz_values = [0]
        next_bf = None
        is_inflection = False

        # 不要キー削除処理
        while fno < fnos[-1]:
            is_inflection = False
            inflection_fno = start_fno

            bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)
            logger.debug("*%s: f: %s, bf(%s):rot:%s", bone_name, fno, bf.fno, bf.rotation.toEulerAngles4MMD().to_log())

            prev_bf = self.calc_bf(bone_name, start_fno, is_key=False, is_read=False, is_reset_interpolation=False)
            logger.debug("*%s: f: %s, prev_bf(%s):rot:%s", bone_name, fno, prev_bf.fno, prev_bf.rotation.toEulerAngles4MMD().to_log())

            next_bf = None

            # 変化量を保持
            rot_values.append(bf.rotation.calcTheata(prev_bf.rotation))
            mx_values.append(bf.position.x() - prev_bf.position.x())
            my_values.append(bf.position.y() - prev_bf.position.y())
            mz_values.append(bf.position.z() - prev_bf.position.z())

            if 2 >= len(rot_values):
                # 必要な個数が溜まるまでスルー
                fno += 1
                continue
            
            if bf.key:
                # 単調増加としてキーを結合してみる
                (joined_rot_bzs, rot_inflection) = MBezierUtils.join_value_2_bezier(fno, bone_name, rot_values, \
                                                                                    offset=offset, diff_limit=rot_diff_limit) if is_rot else (True, [])
                (joined_mx_bzs, mx_inflection) = MBezierUtils.join_value_2_bezier(fno, bone_name, mx_values, \
                                                                                  offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])
                (joined_my_bzs, my_inflection) = MBezierUtils.join_value_2_bezier(fno, bone_name, my_values, \
                                                                                  offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])
                (joined_mz_bzs, mz_inflection) = MBezierUtils.join_value_2_bezier(fno, bone_name, mz_values, \
                                                                                  offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])

                if joined_rot_bzs and joined_mx_bzs and joined_my_bzs and joined_mz_bzs and bf.rotation.toEulerAngles().distanceToPoint(prev_bf.rotation.toEulerAngles()) < 90:
                    next_bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False)

                    # 結合できた場合、補間曲線をnextに設定
                    if is_rot and len(joined_rot_bzs) > 0:
                        logger.debug("☆%s: f: %s(%s), キー:回転補間曲線成功: 1: %s, 2: %s", bone_name, fno, start_fno, joined_rot_bzs[1].to_log(), joined_rot_bzs[2].to_log())
                        self.reset_interpolation_parts(bone_name, next_bf, joined_rot_bzs, MBezierUtils.R_x1_idxs, MBezierUtils.R_y1_idxs, MBezierUtils.R_x2_idxs, MBezierUtils.R_y2_idxs)
                    
                    if is_mov and len(joined_mx_bzs) > 0 and len(joined_my_bzs) > 0 and len(joined_mz_bzs) > 0:
                        logger.debug("☆%s: f: %s(%s), キー:移動X補間曲線成功: 1: %s, 2: %s", bone_name, fno, start_fno, joined_mx_bzs[1].to_log(), joined_mx_bzs[2].to_log())
                        logger.debug("☆%s: f: %s(%s), キー:移動Y補間曲線成功: 1: %s, 2: %s", bone_name, fno, start_fno, joined_my_bzs[1].to_log(), joined_my_bzs[2].to_log())
                        logger.debug("☆%s: f: %s(%s), キー:移動Z補間曲線成功: 1: %s, 2: %s", bone_name, fno, start_fno, joined_mz_bzs[1].to_log(), joined_mz_bzs[2].to_log())
                        self.reset_interpolation_parts(bone_name, next_bf, joined_mx_bzs, MBezierUtils.MX_x1_idxs, MBezierUtils.MX_y1_idxs, MBezierUtils.MX_x2_idxs, MBezierUtils.MX_y2_idxs)
                        self.reset_interpolation_parts(bone_name, next_bf, joined_my_bzs, MBezierUtils.MY_x1_idxs, MBezierUtils.MY_y1_idxs, MBezierUtils.MY_x2_idxs, MBezierUtils.MY_y2_idxs)
                        self.reset_interpolation_parts(bone_name, next_bf, joined_mz_bzs, MBezierUtils.MZ_x1_idxs, MBezierUtils.MZ_y1_idxs, MBezierUtils.MZ_x2_idxs, MBezierUtils.MZ_y2_idxs)
                    
                    last_fno = fno

                    # 次も結合テストするため、分離はしない
                    is_inflection = False
                else:
                    # 結合できなかった場合、変曲点チェックに移る
                    logger.debug("★%s: f: %s(%s), キー:補間曲線失敗: rot_inflection: %s, mx_inflection: %s, my_inflection: %s, mz_inflection: %s", \
                                    bone_name, fno, start_fno, rot_inflection, mx_inflection, my_inflection, mz_inflection)
                    
                    # 前回の分離点で分離する                    
                    is_inflection = True

                    # # 変曲点があった場合、そこで区切る
                    # indices = np.array([])
                    # indices = np.append(indices, rot_inflection)
                    # indices = np.append(indices, mx_inflection)
                    # indices = np.append(indices, my_inflection)
                    # indices = np.append(indices, mz_inflection)

                    # if indices.size > 0:
                    #     # 昇順に並べ替える
                    #     indices.sort()
                    #     logger.debug("indices: %s", indices)
                    #     # 変曲点で区切る
                    #     inflection = int(indices[0])

                    #     if inflection <= 0:
                    #         fno += 1
                    #         continue

                    #     inflection_fno = start_fno + inflection
                    #     logger.debug("☆%s: 変曲点: %s, start_f: %s, f: %s, indices: %s", bone_name, inflection_fno, start_fno, fno, indices)

                    #     # 結合したベジェ曲線
                    #     (joined_rot_bzs, rot_inflection) = MBezierUtils.join_value_2_bezier(inflection_fno, bone_name, rot_values[:(inflection + 1)], \
                    #                                                                         offset=offset, diff_limit=rot_diff_limit) if is_rot else (True, [])
                    #     (joined_mx_bzs, mx_inflection) = MBezierUtils.join_value_2_bezier(inflection_fno, bone_name, mx_values[:(inflection + 1)], \
                    #                                                                       offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])
                    #     (joined_my_bzs, my_inflection) = MBezierUtils.join_value_2_bezier(inflection_fno, bone_name, my_values[:(inflection + 1)], \
                    #                                                                       offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])
                    #     (joined_mz_bzs, mz_inflection) = MBezierUtils.join_value_2_bezier(inflection_fno, bone_name, mz_values[:(inflection + 1)], \
                    #                                                                       offset=offset, diff_limit=mov_diff_limit) if is_mov else (True, [])

                    #     if joined_rot_bzs and joined_mx_bzs and joined_my_bzs and joined_mz_bzs:
                    #         next_bf = self.calc_bf(bone_name, inflection_fno, is_key=False, is_read=False, is_reset_interpolation=False)

                    #         # 結合できた場合、補間曲線をnextに設定
                    #         if is_rot:
                    #             logger.debug("☆%s: f: %s(%s), 変曲点:回転補間曲線成功: 1: %s, 2: %s", bone_name, inflection_fno, start_fno, joined_rot_bzs[1].to_log(), joined_rot_bzs[2].to_log())
                    #             self.reset_interpolation_parts(bone_name, next_bf, joined_rot_bzs, MBezierUtils.R_x1_idxs, MBezierUtils.R_y1_idxs, MBezierUtils.R_x2_idxs, MBezierUtils.R_y2_idxs)
                            
                    #         if is_mov:
                    #             logger.debug("☆%s: f: %s(%s), 変曲点:移動X補間曲線成功: 1: %s, 2: %s", bone_name, inflection_fno, start_fno, joined_mx_bzs[1].to_log(), joined_mx_bzs[2].to_log())
                    #             logger.debug("☆%s: f: %s(%s), 変曲点:移動Y補間曲線成功: 1: %s, 2: %s", bone_name, inflection_fno, start_fno, joined_my_bzs[1].to_log(), joined_my_bzs[2].to_log())
                    #             logger.debug("☆%s: f: %s(%s), 変曲点:移動Z補間曲線成功: 1: %s, 2: %s", bone_name, inflection_fno, start_fno, joined_mz_bzs[1].to_log(), joined_mz_bzs[2].to_log())
                    #             self.reset_interpolation_parts(bone_name, next_bf, joined_mx_bzs, MBezierUtils.MX_x1_idxs, MBezierUtils.MX_y1_idxs, MBezierUtils.MX_x2_idxs, MBezierUtils.MX_y2_idxs)
                    #             self.reset_interpolation_parts(bone_name, next_bf, joined_my_bzs, MBezierUtils.MY_x1_idxs, MBezierUtils.MY_y1_idxs, MBezierUtils.MY_x2_idxs, MBezierUtils.MY_y2_idxs)
                    #             self.reset_interpolation_parts(bone_name, next_bf, joined_mz_bzs, MBezierUtils.MZ_x1_idxs, MBezierUtils.MZ_y1_idxs, MBezierUtils.MZ_x2_idxs, MBezierUtils.MZ_y2_idxs)
                            
                    #         # 最終結合キーを置き換える
                    #         last_fno = inflection_fno
                    #     else:
                    #         # 結合できなかった場合、前回最終結合キーで分離する
                    #         logger.debug("★%s: f: %s(%s), 変曲点:補間曲線失敗: rot_inflection: %s, mx_inflection: %s, my_inflection: %s, mz_inflection: %s", \
                    #                     bone_name, inflection_fno, start_fno, rot_inflection, mx_inflection, my_inflection, mz_inflection)
                    # else:
                    #     # 変曲点がない場合（曲線が補間曲線に合わない場合）、前回最終結合キーで分離する
                    #     logger.debug("変曲点が見つからなかった場合")

                # 変曲点で登録
                if next_bf:
                    if bone_name in self.bones and last_fno in self.bones[bone_name]:
                        self.bones[bone_name][last_fno].key = True
                        self.bones[bone_name][last_fno].interpolation = next_bf.interpolation
                        logger.debug("◇登録 %s: f: %s, next_bf(%s) rot:%s", bone_name, last_fno, next_bf.fno, next_bf.rotation.toEulerAngles4MMD().to_log())
                    else:
                        self.regist_bf(next_bf, bone_name, last_fno, copy_interpolation=True)
                        logger.debug("☆登録 %s: f: %s, next_bf(%s) rot:%s", bone_name, last_fno, next_bf.fno, next_bf.rotation.toEulerAngles4MMD().to_log())
                
                for f in range(start_fno + 1, last_fno):
                    # 結合できた場合、区間内を削除
                    if f in self.bones[bone_name]:
                        logger.debug("☆%s: f: %s, キーフレ削除: df: %s", bone_name, last_fno, f)
                        self.bones[bone_name][f].key = False

                if is_inflection:
                    logger.debug("%s, 開始キーフレ変更前: start_fno: %s, fno: %s, last_fno: %s", bone_name, start_fno, fno, last_fno)

                    # 開始キーフレは、変曲点までずらす
                    fno = last_fno if start_fno != last_fno else last_fno + 1
                    start_fno = fno
                    last_fno = fno

                    # 配列初期化
                    rot_values = [0]
                    mx_values = [0]
                    my_values = [0]
                    mz_values = [0]

                    logger.debug("%s, 開始キーフレ変更後: start_fno: %s, fno: %s, last_fno: %s", bone_name, start_fno, fno, last_fno)

            if fno // 300 > prev_sep_fno and fnos[-1] > 0 and is_show_log:
                if data_set_no == 0:
                    logger.info("-- %sフレーム目:終了(%s％)【不要キー削除 - %s】", fno, round((fno / fnos[-1]) * 100, 3), bone_name)
                else:
                    logger.info("-- %sフレーム目:終了(%s％)【No.%s - 不要キー削除 - %s】", fno, round((fno / fnos[-1]) * 100, 3), data_set_no, bone_name)

                prev_sep_fno = fno // 300

            # 必ず進める
            fno += 1

            if r_start_fno < 0 and r_end_fno < 0:
                # 範囲指定がない場合、全範囲
                active_fnos = self.get_bone_fnos(bone_name, is_key=True)
            else:
                # 範囲指定がある場合はその範囲内だけ
                active_fnos = self.get_bone_fnos(bone_name, start_fno=r_start_fno, end_fno=r_end_fno, is_key=True)

            logger.debug("**☆有効キー(%s-%s) %s: %s", bone_name, start_fno, fno, active_fnos)
        
        logger.debug("len: %s, last: %s", len(rot_values), fnos[-1])

        if len(rot_values) == fnos[-1]:
            # 最後まで変化がない場合、1F目以降を削除
            for f in range(1, fnos[-1] + 1):
                if f in self.bones[bone_name]:
                    del self.bones[bone_name][f]

        if r_start_fno < 0 and r_end_fno < 0:
            # 範囲指定がない場合、全範囲
            active_fnos = self.get_bone_fnos(bone_name, is_key=True)

            for fno in range(fnos[-1]):
                if fno in self.bones[bone_name] and fno not in active_fnos:
                    # 最後に物理削除
                    del self.bones[bone_name][fno]
        else:
            # 範囲指定がある場合はその範囲内だけ
            active_fnos = self.get_bone_fnos(bone_name, start_fno=r_start_fno, end_fno=r_end_fno, is_key=True)

            for fno in range(r_start_fno, r_end_fno):
                if fno in self.bones[bone_name] and fno not in active_fnos:
                    # 最後に物理削除
                    del self.bones[bone_name][fno]
            
        logger.debug("remove_unnecessary_bf after: %s, %s, all: %s", bone_name, active_fnos, len(fnos))

    # 補間曲線分割ありで登録
    def regist_bf(self, bf: VmdBoneFrame, bone_name: str, fno: int, copy_interpolation=False, is_key=True):
        # 登録対象の場合のみ、補間曲線リセットで登録する
        regist_bf = self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=True)
        regist_bf.position = bf.position.copy()
        regist_bf.rotation = bf.rotation.copy()
        regist_bf.org_rotation = bf.org_rotation.copy()
        if copy_interpolation:
            regist_bf.interpolation = cPickle.loads(cPickle.dumps(bf.interpolation, -1))
            
        # キーを登録
        regist_bf.key = is_key
        self.bones[bone_name][fno] = regist_bf
        # 補間曲線を設定（有効なキーのみ）
        prev_fno, next_fno = self.get_bone_prev_next_fno(bone_name, fno=fno, is_key=True)

        prev_bf = self.calc_bf(bone_name, prev_fno, is_key=False, is_read=False, is_reset_interpolation=False)
        next_bf = self.calc_bf(bone_name, next_fno, is_key=False, is_read=False, is_reset_interpolation=False)
        self.split_bf_by_fno(bone_name, prev_bf, next_bf, fno)
    
    def regist_mf(self, mf: VmdMorphFrame, morph_name: str, fno: int):
        if morph_name not in self.morphs:
            self.morphs[morph_name] = {}
        
        self.morphs[morph_name][fno] = mf

    # 補間曲線を考慮した指定フレーム番号の位置
    # https://www55.atwiki.jp/kumiho_k/pages/15.html
    # https://harigane.at.webry.info/201103/article_1.html
    def calc_bf(self, bone_name: str, fno: int, is_key=False, is_read=False, is_reset_interpolation=False):
        fill_bf = VmdBoneFrame(fno)

        if bone_name not in self.bones:
            self.bones[bone_name] = {fno: fill_bf}
            fill_bf.set_name(bone_name)
            return fill_bf
        
        # 条件に合致するフレーム番号を探す
        # is_key: 登録対象のキーを探す
        # is_read: データ読み込み時のキーを探す
        if fno in self.bones[bone_name] and (not is_key or (is_key and self.bones[bone_name][fno].key)) and (not is_read or (is_read and self.bones[bone_name][fno].read)):
            # 合致するキーが見つかった場合、それを返す
            return self.bones[bone_name][fno]
        else:
            # 合致するキーが見つからなかった場合
            if is_key or is_read:
                # 既存キーのみ探している場合はNone
                return None

        # 番号より前のフレーム番号
        before_fnos = [x for x in sorted(self.bones[bone_name].keys()) if (x < fno)]
        # 番号より後のフレーム番号
        after_fnos = [x for x in sorted(self.bones[bone_name].keys()) if (x > fno)]

        if len(after_fnos) == 0 and len(before_fnos) == 0:
            fill_bf.set_name(bone_name)
            return fill_bf

        if len(after_fnos) == 0:
            # 番号より前があって、後のがない場合、前のをコピーして返す
            fill_bf = self.bones[bone_name][before_fnos[-1]].copy()
            fill_bf.fno = fno
            fill_bf.key = False
            fill_bf.read = False
            return fill_bf
        
        if len(before_fnos) == 0:
            # 番号より後があって、前がない場合、後のをコピーして返す
            fill_bf = self.bones[bone_name][after_fnos[0]].copy()
            fill_bf.fno = fno
            fill_bf.key = False
            fill_bf.read = False
            return fill_bf

        prev_bf = self.bones[bone_name][before_fnos[-1]]
        next_bf = self.bones[bone_name][after_fnos[0]]

        # 名前をコピー
        fill_bf.name = prev_bf.name
        fill_bf.bname = prev_bf.bname

        # 補間曲線を元に間を埋める
        # rfunc = profile(self.calc_bf_rot)
        # fill_bf.rotation = rfunc(prev_bf, fill_bf, next_bf)
        # pfunc = profile(self.calc_bf_pos)
        # fill_bf.position = pfunc(prev_bf, fill_bf, next_bf)
        fill_bf.rotation = self.calc_bf_rot(prev_bf, fill_bf, next_bf)
        fill_bf.position = self.calc_bf_pos(prev_bf, fill_bf, next_bf)

        if is_reset_interpolation:
            # 補間曲線再設定の場合、範囲外でも構わないので補間曲線を設定する

            # 回転の分割
            r_x, r_y, r_t, r_bresult, r_aresult, r_before_bz, r_after_bz \
                = MBezierUtils.split_bezier_mmd(next_bf.interpolation[MBezierUtils.R_x1_idxs[3]], next_bf.interpolation[MBezierUtils.R_y1_idxs[3]], \
                                                next_bf.interpolation[MBezierUtils.R_x2_idxs[3]], next_bf.interpolation[MBezierUtils.R_y2_idxs[3]], \
                                                prev_bf.fno, fill_bf.fno, next_bf.fno)
            # 移動Xの分割
            x_x, x_y, x_t, x_bresult, x_aresult, x_before_bz, x_aftex_bz \
                = MBezierUtils.split_bezier_mmd(next_bf.interpolation[MBezierUtils.MX_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MX_y1_idxs[3]], \
                                                next_bf.interpolation[MBezierUtils.MX_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MX_y2_idxs[3]], \
                                                prev_bf.fno, fill_bf.fno, next_bf.fno)
            # 移動Yの分割
            y_x, y_y, y_t, y_bresult, y_aresult, y_before_bz, y_aftey_bz \
                = MBezierUtils.split_bezier_mmd(next_bf.interpolation[MBezierUtils.MY_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MY_y1_idxs[3]], \
                                                next_bf.interpolation[MBezierUtils.MY_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MY_y2_idxs[3]], \
                                                prev_bf.fno, fill_bf.fno, next_bf.fno)
            # 移動Zの分割
            z_x, z_y, z_t, z_bresult, z_aresult, z_before_bz, z_aftez_bz \
                = MBezierUtils.split_bezier_mmd(next_bf.interpolation[MBezierUtils.MZ_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MZ_y1_idxs[3]], \
                                                next_bf.interpolation[MBezierUtils.MZ_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MZ_y2_idxs[3]], \
                                                prev_bf.fno, fill_bf.fno, next_bf.fno)

            # 強制設定
            self.reset_interpolation(bone_name, prev_bf, fill_bf, next_bf, r_before_bz, r_after_bz, \
                                     MBezierUtils.R_x1_idxs, MBezierUtils.R_y1_idxs, MBezierUtils.R_x2_idxs, MBezierUtils.R_y2_idxs)
            self.reset_interpolation(bone_name, prev_bf, fill_bf, next_bf, x_before_bz, x_aftex_bz, \
                                     MBezierUtils.MX_x1_idxs, MBezierUtils.MX_y1_idxs, MBezierUtils.MX_x2_idxs, MBezierUtils.MX_y2_idxs)
            self.reset_interpolation(bone_name, prev_bf, fill_bf, next_bf, y_before_bz, y_aftey_bz, \
                                     MBezierUtils.MY_x1_idxs, MBezierUtils.MY_y1_idxs, MBezierUtils.MY_x2_idxs, MBezierUtils.MY_y2_idxs)
            self.reset_interpolation(bone_name, prev_bf, fill_bf, next_bf, z_before_bz, z_aftez_bz, \
                                     MBezierUtils.MZ_x1_idxs, MBezierUtils.MZ_y1_idxs, MBezierUtils.MZ_x2_idxs, MBezierUtils.MZ_y2_idxs)

        return fill_bf

    # 補間曲線を元に、回転ボーンの値を求める
    def calc_bf_rot(self, prev_bf: VmdBoneFrame, fill_bf: VmdBoneFrame, next_bf: VmdBoneFrame):
        if prev_bf.rotation != next_bf.rotation:
            # 回転補間曲線
            rx, ry, rt = MBezierUtils.evaluate(next_bf.interpolation[MBezierUtils.R_x1_idxs[3]], next_bf.interpolation[MBezierUtils.R_y1_idxs[3]], \
                                               next_bf.interpolation[MBezierUtils.R_x2_idxs[3]], next_bf.interpolation[MBezierUtils.R_y2_idxs[3]], \
                                               prev_bf.fno, fill_bf.fno, next_bf.fno)
            return MQuaternion.slerp(prev_bf.rotation, next_bf.rotation, ry)

        return prev_bf.rotation.copy()

    # 補間曲線を元に移動ボーンの値を求める
    def calc_bf_pos(self, prev_bf: VmdBoneFrame, fill_bf: VmdBoneFrame, next_bf: VmdBoneFrame):
        # 補間曲線を元に間を埋める
        if prev_bf.position != next_bf.position:
            # http://rantyen.blog.fc2.com/blog-entry-65.html
            # X移動補間曲線
            xx, xy, xt = MBezierUtils.evaluate(next_bf.interpolation[MBezierUtils.MX_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MX_y1_idxs[3]], \
                                               next_bf.interpolation[MBezierUtils.MX_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MX_y2_idxs[3]], \
                                               prev_bf.fno, fill_bf.fno, next_bf.fno)
            # Y移動補間曲線
            yx, yy, yt = MBezierUtils.evaluate(next_bf.interpolation[MBezierUtils.MY_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MY_y1_idxs[3]], \
                                               next_bf.interpolation[MBezierUtils.MY_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MY_y2_idxs[3]], \
                                               prev_bf.fno, fill_bf.fno, next_bf.fno)
            # Z移動補間曲線
            zx, zy, zt = MBezierUtils.evaluate(next_bf.interpolation[MBezierUtils.MZ_x1_idxs[3]], next_bf.interpolation[MBezierUtils.MZ_y1_idxs[3]], \
                                               next_bf.interpolation[MBezierUtils.MZ_x2_idxs[3]], next_bf.interpolation[MBezierUtils.MZ_y2_idxs[3]], \
                                               prev_bf.fno, fill_bf.fno, next_bf.fno)

            fill_pos = MVector3D()
            fill_pos.setX(prev_bf.position.x() + ((next_bf.position.x() - prev_bf.position.x()) * xy))
            fill_pos.setY(prev_bf.position.y() + ((next_bf.position.y() - prev_bf.position.y()) * yy))
            fill_pos.setZ(prev_bf.position.z() + ((next_bf.position.z() - prev_bf.position.z()) * zy))
            
            return fill_pos
        
        return prev_bf.position.copy()
    
    # キーフレを指定されたフレーム番号の前後で分割する
    def split_bf_by_fno(self, target_bone_name: str, prev_bf: VmdBoneFrame, next_bf: VmdBoneFrame, fill_fno: int):
        if not (prev_bf.fno < fill_fno < next_bf.fno):
            # 間の分割が出来ない場合、終了
            return False

        # 補間曲線もともに分割する
        fill_bf = self.calc_bf(target_bone_name, fill_fno, is_key=False, is_read=False, is_reset_interpolation=True)
        fill_bf.key = True
        self.bones[target_bone_name][fill_fno] = fill_bf

        # 分割結果
        fill_result = True
        # 前半の分割
        fill_result = self.split_bf(target_bone_name, prev_bf, fill_bf) and fill_result
        # 後半の分割
        fill_result = self.split_bf(target_bone_name, fill_bf, next_bf) and fill_result

        return fill_result

    # キーフレを移動量の中心で分割する
    def split_bf(self, target_bone_name: str, prev_bf: VmdBoneFrame, next_bf: VmdBoneFrame):
        if prev_bf.fno == next_bf.fno:
            # 間の分割が出来ない場合、終了
            return True

        # 回転中点fno
        r_fill_fno = self.get_split_fill_fno(target_bone_name, prev_bf, next_bf, \
                                             MBezierUtils.R_x1_idxs, MBezierUtils.R_y1_idxs, MBezierUtils.R_x2_idxs, MBezierUtils.R_y2_idxs)
        # 移動X中点fno
        x_fill_fno = self.get_split_fill_fno(target_bone_name, prev_bf, next_bf, \
                                             MBezierUtils.MX_x1_idxs, MBezierUtils.MX_y1_idxs, MBezierUtils.MX_x2_idxs, MBezierUtils.MX_y2_idxs)
        # 移動Y中点fno
        y_fill_fno = self.get_split_fill_fno(target_bone_name, prev_bf, next_bf, \
                                             MBezierUtils.MY_x1_idxs, MBezierUtils.MY_y1_idxs, MBezierUtils.MY_x2_idxs, MBezierUtils.MY_y2_idxs)
        # 移動Z中点fno
        z_fill_fno = self.get_split_fill_fno(target_bone_name, prev_bf, next_bf, \
                                             MBezierUtils.MZ_x1_idxs, MBezierUtils.MZ_y1_idxs, MBezierUtils.MZ_x2_idxs, MBezierUtils.MZ_y2_idxs)

        fnos = []
        for fill_fno in [r_fill_fno, x_fill_fno, y_fill_fno, z_fill_fno]:
            if fill_fno and prev_bf.fno < fill_fno < next_bf.fno:
                # fnoがあって範囲内の場合、設定対象でfnoを保持
                fnos.append(fill_fno)

        # 重複なしの昇順リスト
        fnos = list(sorted(list(set(fnos))))

        fill_result = True
        if len(fnos) > 0:
            # 現在処理対象以外にfnoがある場合、その最小地点で前後に分割
            fill_result = self.split_bf_by_fno(target_bone_name, prev_bf, next_bf, fnos[0]) and fill_result
        
        return fill_result
    
    # キーフレを指定bf間の中間で区切れるフレーム番号を取得する
    # 分割が不要（範囲内に収まってる）場合、-1で対象外
    def get_split_fill_fno(self, target_bone_name: str, prev_bf: VmdBoneFrame, next_bf: VmdBoneFrame, \
                           x1_idxs: list, y1_idxs: list, x2_idxs: list, y2_idxs: list):
        next_x1v = next_bf.interpolation[x1_idxs[3]]
        next_y1v = next_bf.interpolation[y1_idxs[3]]
        next_x2v = next_bf.interpolation[x2_idxs[3]]
        next_y2v = next_bf.interpolation[y2_idxs[3]]

        if not MBezierUtils.is_fit_bezier_mmd([MVector2D(), MVector2D(next_x1v, next_y1v), MVector2D(next_x2v, next_y2v), MVector2D()]):
            # ベジェ曲線がMMDの範囲内に収まっていない場合、中点で分割
            new_fill_fno, _, _ = MBezierUtils.evaluate_by_t(next_x1v, next_y1v, next_x2v, next_y2v, prev_bf.fno, next_bf.fno, 0.5)

            if prev_bf.fno < new_fill_fno < next_bf.fno:
                return new_fill_fno
        
        # 範囲内でない場合
        return -1

    # 補間曲線の再設定処理
    def reset_interpolation(self, target_bone_name: str, prev_bf: VmdBoneFrame, now_bf: VmdBoneFrame, next_bf: VmdBoneFrame, \
                            before_bz: list, after_bz: list, x1_idxs: list, y1_idxs: list, x2_idxs: list, y2_idxs: list):
        
        # 今回キーに設定
        self.reset_interpolation_parts(target_bone_name, now_bf, before_bz, x1_idxs, y1_idxs, x2_idxs, y2_idxs)

        # nextキーに設定
        self.reset_interpolation_parts(target_bone_name, next_bf, after_bz, x1_idxs, y1_idxs, x2_idxs, y2_idxs)
    
    # 補間曲線のコピー
    def copy_interpolation(self, org_bf: VmdBoneFrame, rep_bf: VmdBoneFrame, bz_type: str):
        org_interpolation = cPickle.loads(cPickle.dumps(org_bf.interpolation, -1))

        bz_x1_idxs, bz_y1_idxs, bz_x2_idxs, bz_y2_idxs = MBezierUtils.from_bz_type(bz_type)

        rep_bf.interpolation[bz_x1_idxs[0]] = rep_bf.interpolation[bz_x1_idxs[1]] = rep_bf.interpolation[bz_x1_idxs[2]] = rep_bf.interpolation[bz_x1_idxs[3]] \
            = org_interpolation[bz_x1_idxs[3]]
        rep_bf.interpolation[bz_y1_idxs[0]] = rep_bf.interpolation[bz_y1_idxs[1]] = rep_bf.interpolation[bz_y1_idxs[2]] = rep_bf.interpolation[bz_y1_idxs[3]] \
            = org_interpolation[bz_y1_idxs[3]]

        rep_bf.interpolation[bz_x2_idxs[0]] = rep_bf.interpolation[bz_x2_idxs[2]] = rep_bf.interpolation[bz_x2_idxs[2]] = rep_bf.interpolation[bz_x2_idxs[3]] \
            = org_interpolation[bz_x2_idxs[3]]
        rep_bf.interpolation[bz_y2_idxs[0]] = rep_bf.interpolation[bz_y2_idxs[2]] = rep_bf.interpolation[bz_y2_idxs[2]] = rep_bf.interpolation[bz_y2_idxs[3]] \
            = org_interpolation[bz_y2_idxs[3]]

    # 補間曲線の再設定部品
    def reset_interpolation_parts(self, target_bone_name: str, bf: VmdBoneFrame, bzs: list, x1_idxs: list, y1_idxs: list, x2_idxs: list, y2_idxs: list):
        # キーの始点は、B
        bf.interpolation[x1_idxs[0]] = bf.interpolation[x1_idxs[1]] = bf.interpolation[x1_idxs[2]] = bf.interpolation[x1_idxs[3]] = int(bzs[1].x())
        bf.interpolation[y1_idxs[0]] = bf.interpolation[y1_idxs[1]] = bf.interpolation[y1_idxs[2]] = bf.interpolation[y1_idxs[3]] = int(bzs[1].y())

        # キーの終点は、C
        bf.interpolation[x2_idxs[0]] = bf.interpolation[x2_idxs[1]] = bf.interpolation[x2_idxs[2]] = bf.interpolation[x2_idxs[3]] = int(bzs[2].x())
        bf.interpolation[y2_idxs[0]] = bf.interpolation[y2_idxs[1]] = bf.interpolation[y2_idxs[2]] = bf.interpolation[y2_idxs[3]] = int(bzs[2].y())

    # モーフモーション：フレーム番号リスト
    def get_morph_fnos(self, *morph_names, **kwargs):
        if not self.morphs:
            return []
        
        # is_key: 登録対象のキーを探す
        # is_read: データ読み込み時のキーを探す
        is_key = True if "is_key" in kwargs and kwargs["is_key"] else False
        is_read = True if "is_read" in kwargs and kwargs["is_read"] else False
        start_fno = kwargs["start_fno"] if "start_fno" in kwargs and kwargs["start_fno"] else 0
        end_fno = kwargs["end_fno"] if "end_fno" in kwargs and kwargs["end_fno"] else 9999999999
        
        # 条件に合致するフレーム番号を探す
        keys = []
        for morph_name in morph_names:
            if morph_name in self.morphs:
                keys.extend([x for x in self.morphs[morph_name].keys() if self.morphs[morph_name][x].fno >= start_fno and self.morphs[morph_name][x].fno <= end_fno and \
                            (not is_key or (is_key and self.morphs[morph_name][x].key)) and (not is_read or (is_read and self.morphs[morph_name][x].read))])
        
        # 重複を除いた昇順フレーム番号リストを返す
        return sorted(list(set(keys)))

    # モーフ登録
    def regist_mf(self, mf: VmdMorphFrame, morph_name: str, fno: int):
        if math.isnan(mf.ratio) or math.isinf(mf.ratio):
            logger.debug("** regist_mf: (%s)%s", mf.fno, mf.ratio)

        regist_mf = VmdMorphFrame(mf.fno)
        regist_mf.set_name(mf.name)
        regist_mf.ratio = get_effective_value(mf.ratio)

        if morph_name not in self.morphs:
            self.morphs[morph_name] = {}

        if math.isnan(regist_mf.ratio) or math.isinf(regist_mf.ratio):
            logger.debug("*** c_regist_mf: (%s)%s", regist_mf.fno, regist_mf.ratio)

        # キーを登録
        regist_mf.key = True
        self.morphs[morph_name][fno] = regist_mf

    # 指定フレーム番号のモーフ
    def calc_mf(self, morph_name: str, fno: int, is_key=False, is_read=False):
        fill_mf = VmdMorphFrame(fno)

        if morph_name not in self.morphs:
            fill_mf.set_name(morph_name)
            self.morphs[morph_name] = {fno: fill_mf}
            return fill_mf
        
        # 条件に合致するフレーム番号を探す
        # is_key: 登録対象のキーを探す
        # is_read: データ読み込み時のキーを探す
        if fno in self.morphs[morph_name] and (not is_key or (is_key and self.morphs[morph_name][fno].key)) and (not is_read or (is_read and self.morphs[morph_name][fno].read)):
            # 合致するキーが見つかった場合、それを返す
            logger.debug("** find: fill: (%s)%s", fill_mf.fno, self.morphs[morph_name][fno].ratio)
            return self.morphs[morph_name][fno]
        else:
            # 合致するキーが見つからなかった場合
            if is_key or is_read:
                # 既存キーのみ探している場合はNone
                return None

        # 番号より前のフレーム番号
        before_fnos = [x for x in sorted(self.morphs[morph_name].keys()) if (x < fno)]
        # 番号より後のフレーム番号
        after_fnos = [x for x in sorted(self.morphs[morph_name].keys()) if (x > fno)]

        if len(after_fnos) == 0 and len(before_fnos) == 0:
            fill_mf.set_name(morph_name)
            return fill_mf

        if len(after_fnos) == 0:
            # 番号より前があって、後のがない場合、前のをコピーして返す
            fill_mf = self.morphs[morph_name][before_fnos[-1]].copy()
            fill_mf.fno = fno
            fill_mf.key = False
            fill_mf.read = False
            logger.debug("** not after: fill: (%s)%s", fill_mf.fno, fill_mf.ratio)
            return fill_mf
        
        if len(before_fnos) == 0:
            # 番号より後があって、前がない場合、後のをコピーして返す
            fill_mf = self.morphs[morph_name][after_fnos[0]].copy()
            fill_mf.fno = fno
            fill_mf.key = False
            fill_mf.read = False
            logger.debug("** not before: fill: (%s)%s", fill_mf.fno, fill_mf.ratio)
            return fill_mf

        prev_mf = self.morphs[morph_name][before_fnos[-1]]
        next_mf = self.morphs[morph_name][after_fnos[0]]
        if math.isnan(prev_mf.ratio) or math.isinf(prev_mf.ratio):
            logger.debug("** prev_mf: (%s)%s", prev_mf.fno, prev_mf.ratio)
        if math.isnan(next_mf.ratio) or math.isinf(next_mf.ratio):
            logger.debug("** next_mf: (%s)%s", next_mf.fno, next_mf.ratio)

        # 名前をコピー
        fill_mf.name = prev_mf.name
        fill_mf.bname = prev_mf.bname

        # 線形で埋める
        # rx, ry, rt = MBezierUtils.evaluate(MBezierUtils.LINEAR_MMD_INTERPOLATION[1].x(), MBezierUtils.LINEAR_MMD_INTERPOLATION[1].y(), \
        #                                     MBezierUtils.LINEAR_MMD_INTERPOLATION[2].x(), MBezierUtils.LINEAR_MMD_INTERPOLATION[2].y(), \
        #                                     prev_mf.fno, fill_mf.fno, next_mf.fno)
        # fill_mf.ratio = prev_mf.ratio + ((next_mf.ratio - prev_mf.ratio) * ry)
        fill_mf.ratio = prev_mf.ratio + ((next_mf.ratio - prev_mf.ratio) * ((fill_mf.fno - prev_mf.fno) / (next_mf.fno - prev_mf.fno)))
        logger.debug("** fill: (%s)%s, head: (%s)%s, tail: (%s)%s", fill_mf.fno, fill_mf.ratio, prev_mf.fno, prev_mf.ratio, next_mf.fno, next_mf.ratio)
        if math.isnan(fill_mf.ratio) or math.isinf(fill_mf.ratio):
            logger.debug("** fill: (%s)%s, head_mf: %s, %s tail_mf: %s, %s", fill_mf.fno, fill_mf.ratio, prev_mf.fno, prev_mf.ratio, next_mf.fno, next_mf.ratio)

        return fill_mf

    def smooth_filter_mf(self, data_set_no: int, morph_name: str, loop=1, \
                         config={"freq": 30, "mincutoff": 0.3, "beta": 0.01, "dcutoff": 0.25}, start_fno=-1, end_fno=-1, is_show_log=True):
        prev_sep_fno = 0

        for n in range(loop):
            rxfilter = OneEuroFilter(**config)

            fnos = self.get_morph_fnos(morph_name)
            prev_sep_fno = 0

            # キーフレを取得する
            if start_fno < 0 and end_fno < 0:
                # 範囲指定がない場合、全範囲
                fnos = self.get_morph_fnos(morph_name)
            else:
                # 範囲指定がある場合はその範囲内だけ
                fnos = self.get_morph_fnos(morph_name, start_fno=start_fno, end_fno=end_fno)

            # 全区間をフィルタにかける
            for fno in fnos:
                now_mf = self.calc_mf(morph_name, fno, is_key=False, is_read=False)
                now_mf.ratio = rxfilter(now_mf.ratio, fno)

                if is_show_log and data_set_no > 0 and fno // 2000 > prev_sep_fno and fnos[-1] > 0:
                    logger.info("-- %sフレーム目:終了(%s％)【No.%s - フィルタリング - %s(%s)】", fno, round((fno / fnos[-1]) * 100, 3), data_set_no, morph_name, (n + 1))
                    prev_sep_fno = fno // 2000

    # 無効なキーを物理削除する
    def remove_unkey_mf(self, data_set_no: int, morph_name: str):
        for fno in self.get_morph_fnos(morph_name):
            mf = self.calc_mf(morph_name, fno, is_key=False, is_read=False)

            if fno in self.morphs[morph_name] and not mf.key:
                del self.morphs[morph_name][fno]

    # 指定モーフの不要キーを削除する
    # 変曲点を求める
    # https://teratail.com/questions/162391
    def remove_unnecessary_mf(self, data_set_no: int, morph_name: str, threshold=0.05, diff_limit=0.01, r_start_fno=-1, r_end_fno=-1, is_show_log=True, is_force=False):
        # キーフレを取得する
        if r_start_fno < 0 and r_end_fno < 0:
            # 範囲指定がない場合、全範囲
            fnos = self.get_morph_fnos(morph_name)
        else:
            # 範囲指定がある場合はその範囲内だけ
            fnos = self.get_morph_fnos(morph_name, start_fno=r_start_fno, end_fno=r_end_fno)
        logger.debug("reratioe_unnecessary_mf fnos: %s, %s", morph_name, fnos)
        
        if len(fnos) <= 1:
            return
        
        reduce_fnos = self.reduce_morph_frame(morph_name, fnos, fnos[0], fnos[-1], threshold=threshold)
        reduce_fnos.append(fnos[-1])

        for f in fnos:
            if f not in reduce_fnos and f in self.morphs[morph_name]:
                # キーフレが残す対象でない場合、削除
                del self.morphs[morph_name][f]
        
    # キーフレームを間引く
    # オリジナル：https://github.com/errno-mmd/smoothvmd/blob/master/reducevmd.cc
    def reduce_morph_frame(self, morph_name: str, fnos: list, head: int, tail: int, threshold: float):
        # ratioのエラー最大値
        max_err = float(0.0)
        # ratio：エラー最大値のindex
        max_idx = 0

        # 開始のモーフ
        head_mf = self.calc_mf(morph_name, head, is_key=False, is_read=False)
        # 終了のモーフ
        tail_mf = self.calc_mf(morph_name, tail, is_key=False, is_read=False)
        logger.debug("head_mf: %s, %s tail_mf: %s, %s", head_mf.fno, head_mf.ratio, tail_mf.fno, tail_mf.ratio)
        if math.isnan(head_mf.ratio) or math.isinf(head_mf.ratio):
            logger.debug("head_mf: %s, %s tail_mf: %s, %s", head_mf.fno, head_mf.ratio, tail_mf.fno, tail_mf.ratio)

        for i in range(head + 1, tail, 1):
            # rx, ry, rt = MBezierUtils.evaluate(MBezierUtils.LINEAR_MMD_INTERPOLATION[1].x(), MBezierUtils.LINEAR_MMD_INTERPOLATION[1].y(), \
            #                                     MBezierUtils.LINEAR_MMD_INTERPOLATION[2].x(), MBezierUtils.LINEAR_MMD_INTERPOLATION[2].y(), \
            #                                     head_mf.fno, i, tail_mf.fno)
            # ip_ratio = head_mf.ratio + ((tail_mf.ratio - head_mf.ratio) * ry)
            ip_ratio = head_mf.ratio + ((tail_mf.ratio - head_mf.ratio) * ((i - head_mf.fno) / (tail_mf.fno - head_mf.fno)))
            # ip_ratio = head_mf.ratio + (tail_mf.ratio - head_mf.ratio) * (i - head) / total
            now_mf = self.calc_mf(morph_name, i, is_key=False, is_read=False)
            logger.debug("morph_name: %s, i: %s ip_ratio: %s, now: %s, head: (%s)%s, tail: (%s)%s", morph_name, i, ip_ratio, now_mf.ratio, head_mf.fno, head_mf.ratio, tail_mf.fno, tail_mf.ratio)
            pos_err = abs(ip_ratio - now_mf.ratio)

            if pos_err > max_err:
                max_idx = i
                max_err = pos_err

        logger.debug("max_err: %s max_idx: %s", max_err, max_idx)

        v1 = []
        if max_err > threshold:
            v1 = self.reduce_morph_frame(morph_name, fnos, head, max_idx, threshold)
            v2 = self.reduce_morph_frame(morph_name, fnos, max_idx, tail, threshold)
            logger.debug("threshold v1: %s", v1)
            logger.debug("threshold v2: %s", v2)

            v1.extend(v2)
        else:
            v1.append(head_mf.fno)
            logger.debug("not threshold v1: %s", head_mf.fno)

        return v1

    # 有効なキーフレが入っているか
    def is_active_bones(self, bone_name: str):
        for bf in self.bones[bone_name].values():
            if bf.position != MVector3D():
                return True
            if bf.rotation != MQuaternion():
                return True
        
        return False

    # ボーンモーション：フレーム番号リスト
    def get_bone_fnos(self, *bone_names, **kwargs):
        if not self.bones:
            return []
        
        # is_key: 登録対象のキーを探す
        # is_read: データ読み込み時のキーを探す
        is_key = True if "is_key" in kwargs and kwargs["is_key"] else False
        is_read = True if "is_read" in kwargs and kwargs["is_read"] else False
        start_fno = kwargs["start_fno"] if "start_fno" in kwargs and kwargs["start_fno"] else 0
        end_fno = kwargs["end_fno"] if "end_fno" in kwargs and kwargs["end_fno"] else 9999999999
        
        # 条件に合致するフレーム番号を探す
        keys = []
        for bone_name in bone_names:
            if bone_name in self.bones:
                keys.extend([x for x in self.bones[bone_name].keys() if self.bones[bone_name][x].fno >= start_fno and self.bones[bone_name][x].fno <= end_fno and \
                            (not is_key or (is_key and self.bones[bone_name][x].key)) and (not is_read or (is_read and self.bones[bone_name][x].read))])
        
        # 重複を除いた昇順フレーム番号リストを返す
        return sorted(list(set(keys)))
    
    # 指定されたfnoの前後のキーを取得する
    def get_bone_prev_next_fno(self, *bone_names, **kwargs):
        # 指定されたボーン名のfnos
        fnos = self.get_bone_fnos(*bone_names, **kwargs)

        fno = kwargs["fno"] if "fno" in kwargs else 0

        # 指定より前のキーフレ
        prev_fnos = [x for x in fnos if x < fno]
        # 指定より後のキーフレ
        next_fnos = [x for x in fnos if x > fno]
        
        # 前のは取れなければ-1で強制的に前の
        prev_fno = -1 if len(prev_fnos) <= 0 else prev_fnos[-1]
        # 後のは取れなければ最終フレーム＋1
        next_fno = self.last_motion_frame + 1 if len(next_fnos) <= 0 else next_fnos[0]

        return prev_fno, next_fno

    # モーフモーション：フレーム番号リスト
    def get_morph_fnos(self, morph_name: str):
        if not self.morphs or morph_name not in self.morphs:
            return []
        
        return sorted([fno for fno in self.morphs[morph_name].keys()])

    # カメラモーション：フレーム番号リスト
    def get_camera_fnos(self):
        if not self.cameras:
            return []
        
        return sorted([fno for fno in self.cameras.keys()])
        
    # ボーンモーション：一次元配列
    def get_bone_frames(self):
        total_bone_frames = []

        target_fnos = {}

        for bone_name, bone_frames in self.bones.items():
            if bone_name not in ["SIZING_ROOT_BONE", "頭頂", "右つま先実体", "左つま先実体", "右足底辺", "左足底辺", "右足底実体", "左足底実体", "右足ＩＫ底実体", "左足ＩＫ底実体", "右足IK親底実体", "左足IK親底実体", \
                                 "首根元", "右腕下延長", "左腕下延長", "右腕垂直", "左腕垂直", "センター実体", "左腕ひじ中間", "右腕ひじ中間", "左ひじ手首中間", "右ひじ手首中間", "左手首実体", "右手首実体", \
                                 "左親指先実体", "左人指先実体", "左中指先実体", "左薬指先実体", "左小指先実体", "右親指先実体", "右人指先実体", "右中指先実体", "右薬指先実体", "右小指先実体"]:
                # サイジング用ボーンは出力しない
                target_fnos[bone_name] = self.get_bone_fnos(bone_name, is_key=True)

        for bone_name, fnos in target_fnos.items():
            logger.test("%s, %s", bone_name, target_fnos[bone_name])

            if len(fnos) > 0:
                # 各ボーンの最終キーだけ先に登録
                total_bone_frames.append(self.bones[bone_name][fnos[-1]])
        
        for bone_name, fnos in target_fnos.items():
            if len(fnos) > 1:
                # キーフレを最後の一つ手前まで登録
                for fno in fnos[:-1]:
                    if self.bones[bone_name][fno].key:
                        total_bone_frames.append(self.bones[bone_name][fno])
        
        return total_bone_frames
    
    # モーフモーション：一次元配列
    def get_morph_frames(self):
        total_morph_frames = []

        for morph_name, morph_frames in self.morphs.items():
            fnos = self.get_morph_fnos(morph_name)
            
            if len(fnos) > 0:
                # 各モーフの最終キーだけ先に登録
                total_morph_frames.append(morph_frames[fnos[-1]])
        
        for morph_name, morph_frames in self.morphs.items():
            fnos = self.get_morph_fnos(morph_name)

            if len(fnos) > 1:
                # キーフレを最後の一つ手前まで登録
                for fno in fnos[:-1]:
                    total_morph_frames.append(morph_frames[fno])
        
        return total_morph_frames

    # カメラモーション：一次元配列
    def get_camera_frames(self):
        total_camera_frames = []

        # カメラキーフレを逆順に登録
        for fno in reversed(self.get_camera_fnos()):
            total_camera_frames.append(self.cameras[fno])

        return total_camera_frames

    # ボーンキーフレを追加
    def append_bone_frame(self, frame: VmdBoneFrame):
        if frame.name not in self.bones:
            # まだ該当ボーン名がない場合、追加
            self.bones[frame.name] = {}
        
        self.bones[frame.name][frame.fno] = frame

    # モーフキーフレを追加
    def append_morph_frame(self, frame: VmdMorphFrame):
        if frame.name not in self.morphs:
            # まだ該当モーフ名がない場合、追加
            self.morphs[frame.name] = {}
        
        self.morphs[frame.name][frame.fno] = frame

    # 指定fnoのみのモーションデータを生成する
    def copy_bone_motion(self, fno: int):
        new_motion = VmdMotion()

        for bone_name in self.bones.keys():
            new_motion.bones[bone_name] = {fno: self.calc_bf(bone_name, fno, is_key=False, is_read=False, is_reset_interpolation=False).copy()}
        
        return new_motion

    def copy(self):
        motion = VmdMotion()

        motion.path = cPickle.loads(cPickle.dumps(self.path, -1))
        motion.signature = cPickle.loads(cPickle.dumps(self.signature, -1))
        motion.model_name = cPickle.loads(cPickle.dumps(self.model_name, -1))
        motion.last_motion_frame = cPickle.loads(cPickle.dumps(self.last_motion_frame, -1))
        motion.motion_cnt = cPickle.loads(cPickle.dumps(self.motion_cnt, -1))

        for bone_name, bf_dict in self.bones.items():
            motion.bones[bone_name] = {}
            for bf in bf_dict.values():
                motion.bones[bone_name][bf.fno] = bf.copy()

        motion.morph_cnt = cPickle.loads(cPickle.dumps(self.morph_cnt, -1))
        motion.morphs = cPickle.loads(cPickle.dumps(self.morphs, -1))
        motion.camera_cnt = cPickle.loads(cPickle.dumps(self.camera_cnt, -1))
        motion.cameras = cPickle.loads(cPickle.dumps(self.cameras, -1))

        motion.light_cnt = cPickle.loads(cPickle.dumps(self.light_cnt, -1))
        motion.lights = cPickle.loads(cPickle.dumps(self.lights, -1))
        motion.shadow_cnt = cPickle.loads(cPickle.dumps(self.shadow_cnt, -1))
        motion.shadows = cPickle.loads(cPickle.dumps(self.shadows, -1))
        motion.ik_cnt = cPickle.loads(cPickle.dumps(self.ik_cnt, -1))
        motion.showiks = cPickle.loads(cPickle.dumps(self.showiks, -1))
        
        motion.digest = cPickle.loads(cPickle.dumps(self.digest, -1))

        return motion
