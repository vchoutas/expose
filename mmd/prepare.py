# -*- coding: utf-8 -*-
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

from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info("動画準備開始", decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.video_file):
            logger.error("指定されたファイルパスが存在しません。\n{0}", args.video_file, decoration=MLogger.DECORATION_BOX)
            return False, None

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.video_file).parent) if not args.parent_dir else args.parent_dir
        video = cv2.VideoCapture(args.video_file)

        # 幅
        W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 高さ
        H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 総フレーム数
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps
        fps = video.get(cv2.CAP_PROP_FPS)

        logger.info("【初回チェック】\n　ファイル名: {0}, ファイルサイズ: {1}, 横: {2}, 縦: {3}, フレーム数: {4}, fps: {5}", \
                    args.video_file, os.path.getsize(args.video_file), W, H, count, fps, decoration=MLogger.DECORATION_BOX)

        # 縮尺を調整(Colabは容量の問題でちょっと小さめ)
        if W > H:
            # 横長の場合
            width = int(1280) if args.parent_dir else int(1920)
        else:
            # 縦長の場合
            width = int(720) if args.parent_dir else int(1080)

        if len(args.parent_dir) > 0:
            process_img_dir = base_path
        else:
            process_img_dir = os.path.join(base_path, "{0}_{1:%Y%m%d_%H%M%S}".format(os.path.basename(args.video_file).replace('.', '_'), datetime.datetime.now()))

        # 既存は削除
        if os.path.exists(process_img_dir):
            shutil.rmtree(process_img_dir)

        # フォルダ生成
        os.makedirs(process_img_dir)
        os.makedirs(os.path.join(process_img_dir, "resize"), exist_ok=True)
        os.makedirs(os.path.join(process_img_dir, "frames"), exist_ok=True)

        # リサイズpng出力先
        resize_img_path = os.path.join(process_img_dir, "resize", "resize_{0:012}.png")
        # 補間png出力先
        process_img_path = os.path.join(process_img_dir, "frames", "{0:012}", "frame_{0:012}.png")

        # 縮尺
        scale = width / W

        # 縮尺後の高さ
        height = int(H * scale)

        if width % 40 != 0 or height % 40 != 0:
            logger.error("入力動画のサイズが調整後に40で割り切れません。調整前({0}x{1}) -> 調整後({2}-{3})\n40の倍数で入力動画のサイズを調整してください。\n{0}", 
                         W, H, width, height, args.video_file, decoration=MLogger.DECORATION_BOX)
            return False, None

        try:
            # 入力ファイル
            cap = cv2.VideoCapture(args.video_file)

            logger.info("元動画読み込み開始", decoration=MLogger.DECORATION_BOX)

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
                        img = img.resize((width, height), Image.ANTIALIAS)
                        
                        # # floatに変換
                        # img = img_as_float(img)

                        # # コントラストを強調
                        # img = exposure.equalize_hist(img)

                        # # TV filter
                        # img = restoration.denoise_tv_chambolle(img, weight=0.1)

                    except Exception as e:
                        # エラーするようなら無視
                        logger.error(e)

                    # opencv用に変換
                    out_frame = img_as_ubyte(img)
                    # out_frame = np.array(img, dtype=np.uint8)

                    # PNG出力
                    cv2.imwrite(resize_img_path.format(n), out_frame)

            # 補間 --------------------------

            logger.info("補間生成開始", decoration=MLogger.DECORATION_BOX)

            # 元のフレームを30fpsで計算し直した場合の1Fごとの該当フレーム数
            interpolations = np.arange(0, count + 1, fps / 30)

            for kidx, k in enumerate(tqdm(interpolations)):
                # 30fps用にディレクトリ作成
                os.makedirs(os.path.join(process_img_dir, "frames", f"{kidx:012}"), exist_ok=True)

                # コピー対象の画像パス
                target_path = resize_img_path.format(round(k))

                if not os.path.exists(target_path):
                    # 最終フレームとかで対象パスがない場合、ひとつ手前
                    target_path = resize_img_path.format(round(k) - 1)

                process_path = process_img_path.format(kidx)
                if not os.path.exists(target_path):
                    # 最終フレームとかで対象パスがない場合、ひとつ手前
                    target_path = process_img_path.format(kidx - 1)

                # 該当フレーム番号の画像をコピー
                shutil.copy(target_path, process_path)

            # 終わったら開放
            cap.release()

            logger.info("【再チェック】\n　準備フォルダ: {0}, 横: {1}, 縦: {2}, フレーム数: {3}, fps: {4}", process_img_dir, width, height, round(interpolations[-1]), 30)
        except Exception as e:
            logger.error("再エンコード失敗", e)
            return False, None

        cv2.destroyAllWindows()

        # resizeは削除
        shutil.rmtree(os.path.join(process_img_dir, "resize"))

        logger.info("動画準備完了: {0}", process_img_dir, decoration=MLogger.DECORATION_BOX)

        return True, process_img_dir
    except Exception as e:
        logger.critical("動画準備で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None


