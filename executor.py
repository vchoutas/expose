# -*- coding: utf-8 -*-
import argparse
import time
import os

from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)


def show_worked_time(elapsed_time):
    # 経過秒数を時分秒に変換
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "{0:02d}秒".format(int(td_s))
    elif td_h == 0:
        worked_time = "{0:02d}分{1:02d}秒".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}時間{1:02d}分{2:02d}秒".format(int(td_h), int(td_m), int(td_s))

    return worked_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-file', type=str, dest='video_file', default='', help='Video file path')
    parser.add_argument('--parent-dir', type=str, dest='parent_dir', default='', help='Process parent dir path')
    parser.add_argument('--process', type=str, dest='process', default='', help='Process to be executed')
    parser.add_argument('--img-dir', type=str, dest='img_dir', default='', help='Prepared image directory')
    parser.add_argument('--audio-file', type=str, dest='audio_file', default='', help='Audio file path')
    parser.add_argument('--tracking-config', type=str, dest='tracking_config', default="config/tracking-config.yaml", help='Learning model for person tracking')
    parser.add_argument('--tracking-model', type=str, dest='tracking_model', default="lighttrack/weights/mobile-deconv/snapshot_296.ckpt", help='Learning model for person tracking')
    parser.add_argument('--face-model', type=str, dest='face_model', default="data/shape_predictor_68_face_landmarks.dat", help='Learning model for person face')
    parser.add_argument('--order-file', type=str, dest='order_file', default='', help='Index ordering file path')
    parser.add_argument('--bone-config', type=str, dest='bone_config', default="config/あにまさ式ミク準標準ボーン.csv", help='MMD Model Bone csv')
    parser.add_argument('--trace-mov-model-config', type=str, dest='trace_mov_model_config', default="config/trace_mov_model.pmx", help='MMD Model Bone pmx')
    parser.add_argument('--trace-rot-model-config', type=str, dest='trace_rot_model_config', default="config/trace_rot_model.pmx", help='MMD Model Bone pmx')
    parser.add_argument('--hand-motion', type=int, dest='hand_motion', default="0", help='Whether to generate hand motion')
    parser.add_argument('--face-motion', type=int, dest='face_motion', default="0", help='Whether to generate face motion')
    parser.add_argument('--only-json', type=int, dest='only_json', default="0", help='Output Only Json')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='Log level')
    parser.add_argument("--log-mode", type=int, dest='log_mode', default=0, help='Log output mode')

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode)
    result = True

    start = time.time()

    logger.info("MMD自動トレース開始\n　処理対象映像ファイル: {0}\n　処理内容: {1}", args.video_file, args.process, decoration=MLogger.DECORATION_BOX)

    if "prepare" in args.process:
        # 準備
        import mmd.prepare
        result, args.img_dir = mmd.prepare.execute(args)

    if result and "expose" in args.process:
        # exposeによる人物推定
        import mmd.expose
        result = mmd.expose.execute(args)

    if result and "tracking" in args.process:
        # lighttrackによる人物追跡
        import mmd.tracking
        result = mmd.tracking.execute(args)

    if result and "order" in args.process:
        # 人物追跡順番設定
        import mmd.order
        result = mmd.order.execute(args)

    if result and "mediapipe" in args.process:
        # mediapipe推定
        import mmd.mediapipe
        result = mmd.mediapipe.execute(args)

    if result and "face" in args.process:
        # 人物表情推定
        import mmd.face
        result = mmd.face.execute(args)

    if result and "smooth" in args.process:
        # 人物スムージング
        import mmd.smooth
        result = mmd.smooth.execute(args)

    if result and "motion" in args.process:
        # モーション生成
        import mmd.motion
        result = mmd.motion.execute(args)

    elapsed_time = time.time() - start

    logger.info("MMD自動トレース終了\n　処理対象映像ファイル: {0}\n　処理内容: {1}\n　トレース結果: {2}\n　処理時間: {3}", \
                args.video_file, args.process, args.img_dir, show_worked_time(elapsed_time), decoration=MLogger.DECORATION_BOX)

    # 終了音を鳴らす
    if os.name == "nt":
        # Windows
        try:
            import winsound
            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        except Exception:
            pass

