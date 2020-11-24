# -*- coding: utf-8 -*-
import argparse

from mmd.utils.MLogger import MLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-file', type=str, dest='video_file', help='Video file path')
    parser.add_argument('--parent-dir', type=str, dest='parent_dir', default='', help='Process parent dir path')
    parser.add_argument('--process', type=str, dest='process', default='', help='Process to be executed')
    parser.add_argument('--img-dir', type=str, dest='img_dir', default='', help='Prepared image directory')
    parser.add_argument('--tracking-config', type=str, dest='tracking_config', default="config/tracking-config.yaml", help='Learning model for person tracking')
    parser.add_argument('--tracking-model', type=str, dest='tracking_model', default="lighttrack/weights/mobile-deconv/snapshot_296.ckpt", help='Learning model for person tracking')
    parser.add_argument('--order-file', type=str, dest='order_file', default='', help='Index ordering file path')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='Log level')
    parser.add_argument("--log-mode", type=int, dest='log_mode', default=0, help='Log output mode')

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode)
    result = True

    if "prepare" in args.process:
        # 準備
        import mmd.prepare
        result, args.img_dir = mmd.prepare.execute(args)

    if result and "tracking" in args.process:
        # bbox単位の人物追跡
        import mmd.track_bbox
        result = mmd.track_bbox.execute(args)

    if result and "order" in args.process:
        # bbox単位の人物追跡順番設定
        import mmd.order_bbox
        result = mmd.order_bbox.execute(args)

    if result and "expose" in args.process:
        # exposeによる人物推定
        import mmd.expose_bone
        result = mmd.expose_bone.execute(args)
    

