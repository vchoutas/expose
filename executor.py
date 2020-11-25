# -*- coding: utf-8 -*-
import argparse
import time

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
    parser.add_argument('--tracking-config', type=str, dest='tracking_config', default="config/tracking-config.yaml", help='Learning model for person tracking')
    parser.add_argument('--tracking-model', type=str, dest='tracking_model', default="lighttrack/weights/mobile-deconv/snapshot_296.ckpt", help='Learning model for person tracking')
    parser.add_argument('--order-file', type=str, dest='order_file', default='', help='Index ordering file path')
    parser.add_argument('--bone-config', type=str, dest='bone_config', default="config/あにまさ式ミク準標準ボーン.csv", help='MMD Model Bone csv')
    parser.add_argument('--hand-motion', type=int, dest='hand_motion', default="0", help='Whether to generate hand motion')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='Log level')
    parser.add_argument("--log-mode", type=int, dest='log_mode', default=0, help='Log output mode')

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode)
    result = True

    start = time.time()

    logger.info("MMD自動トレース開始\n　処理対象映像ファイル: %s\n　処理対象: %s", args.video_file, args.process, decoration=MLogger.DECORATION_BOX)

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

    if result and "motion" in args.process:
        # モーション生成
        import mmd.motion
        result = mmd.motion.execute(args)

    elapsed_time = time.time() - start

    logger.info("MMD自動トレース終了\n　トレース結果: %s\n　処理時間: %s", args.video_file, show_worked_time(elapsed_time), decoration=MLogger.DECORATION_BOX)
    
