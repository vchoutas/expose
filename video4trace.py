# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import datetime
import math
from tqdm import tqdm
from PIL import Image
import traceback
import numpy as np
import shutil
import re
import pathlib

from skimage import exposure, restoration
from skimage.color import rgb2gray
from skimage import io, exposure, img_as_float, img_as_ubyte
import warnings

from miu.utils.MLogger import MLogger
import miu.config as mconfig

logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    logger.info("MMD自動トレース動画準備開始", decoration=MLogger.DECORATION_LINE)

    if not os.path.exists(args.video_path):
        logger.error("指定されたファイルパスが存在しません。\n(The specified file path does not exist.): {0}".format(args.video_path))
        return

    # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
    base_path = str(pathlib.Path(args.video_path).parent) if not args.parent_dir else args.parent_dir
    video = cv2.VideoCapture(args.video_path)

    # 幅
    W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 高さ
    H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 総フレーム数
    count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # fps
    fps = video.get(cv2.CAP_PROP_FPS)

    logger.info("【初回チェック】\n　ファイル名(file name): {0}, ファイルサイズ(file size): {1}\n".format(args.video_path, os.path.getsize(args.video_path)) \
                + "　横(width): {0}, 縦(height): {1}, フレーム数(count of frames): {2}, fps: {3}".format(W, H, count, fps))

    # 縮尺を調整
    width = int(1280)
    height = int(H * (1280 / W))

    process_img_dir = os.path.join(base_path, mconfig.PROCESS_IMG_DIR, os.path.basename(args.video_path).replace('.', '_'), "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now()))

    # 既存は削除
    if os.path.exists(process_img_dir):
        shutil.rmtree(process_img_dir)

    # フォルダ生成
    os.makedirs(process_img_dir)
    os.makedirs(os.path.join(process_img_dir, "resize"), exist_ok=True)

    # リサイズpng出力先
    resize_img_path = os.path.join(process_img_dir, "resize", "resize_{0:012}.png")
    # 補間png出力先
    process_img_path = os.path.join(process_img_dir, "{0:012}", "frame_{0:012}.png")

    # 縮尺
    scale = width / W

    # オリジナル高さ
    im_height = int(H * scale)

    try:
        # 入力ファイル
        cap = cv2.VideoCapture(args.video_path)

        logger.info("元動画読み込み開始", decoration=MLogger.DECORATION_LINE)

        for n in tqdm(range(int(count))):
            # 動画から1枚キャプチャして読み込む
            flag, img = cap.read()  # Capture frame-by-frame

            # 動画が終わっていたら終了
            if flag == False:
                break
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                try:
                    # 画像に再変換
                    img = Image.fromarray(img)

                    # 画像の縦横を指定サイズに変形
                    img = img.resize((width, im_height), Image.ANTIALIAS)
                    
                    # floatに変換
                    img = img_as_float(img)

                    # # コントラストを強調
                    # img = exposure.equalize_hist(img)

                except Exception as e:
                    # エラーするようなら無視
                    logger.error(e)

                # opencv用に変換
                out_frame = img_as_ubyte(img)  # np.array(img_gray*255, dtype=np.uint8)

                # PNG出力
                cv2.imwrite(resize_img_path.format(n), out_frame)

        logger.info("元動画読み込み終了", decoration=MLogger.DECORATION_LINE)

        # 補間 --------------------------
        # フレーム補間用比率
        fps_interpolation = fps / 30

        logger.info("補間生成開始", decoration=MLogger.DECORATION_LINE)

        # 最後の１つ手前（補間ができる状態）までループ
        for k in tqdm(range(round(count * (30 / fps)) - 1)):
            # 30fps用にディレクトリ作成
            os.makedirs(os.path.join(process_img_dir, f"{k:012}"), exist_ok=True)

            # 補間した出力CNT
            inter_cnt = k * fps_interpolation
            # INDEXと比率（整数部と小数部）
            inter_cnt_rate, inter_cnt_idx = math.modf(inter_cnt)
            # logger.debug("フレーム補間: %s -> %s, idx: %s, rate: %s" % ( cnt, inter_cnt, inter_cnt_idx, inter_cnt_rate ))

            # 前のフレーム
            past_frame = cv2.imread(resize_img_path.format(int(inter_cnt_idx)))
            # 今回のフレーム
            now_frame = cv2.imread(resize_img_path.format(int(inter_cnt_idx + 1)))

            # 混ぜ合わせる比率
            past_rate = inter_cnt_rate
            now_rate = 1 - inter_cnt_rate

            # フレーム補間をして出力する
            target_output_frame = cv2.addWeighted(past_frame, past_rate, now_frame, now_rate, 0)

            # PNG出力
            cv2.imwrite(process_img_path.format(k), target_output_frame)

            # if k % 100 == 0:
            #     logger.info(f"-- 補間生成中 {k}")

        # 最後にnowを出力
        cv2.imwrite(process_img_path.format(k), now_frame)

        logger.info("補間生成終了", decoration=MLogger.DECORATION_LINE)

        # 終わったら開放
        cap.release()
    except Exception as e:
        print("再エンコード失敗", e)
        print(traceback.format_exc())

    cv2.destroyAllWindows()

    logger.info(f'MMD自動トレース動画準備完了: {process_img_dir}', decoration=MLogger.DECORATION_BOX)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-path', type=str, dest='video_path', help='video path')
    parser.add_argument('--parent-dir', type=str, dest='parent_dir', help='process parent dir path', default="")
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')
    
    args = parser.parse_args()

    MLogger.initialize(level=args.verbose, is_file=True)

    execute(args)
