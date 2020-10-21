import argparse
import sys
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
import glob
import pathlib
from miu.utils.MLogger import MLogger

logger = MLogger(__name__, level=MLogger.DEBUG)

RESIZE_PATH = "output/process/resize_img/"
INTERPOLATION_PATH = "output/process/interpolation_img/"

def sort_by_numeric(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def execute(args):
    logger.info("自動トレース用動画変換", decoration=MLogger.DECORATION_LINE)

    if not os.path.exists(args.video_path):
        logger.error("指定されたファイルパスが存在しません。\n(The specified file path does not exist.): {0}".format(args.video_path))
        return

    # 親パス
    base_path = str(pathlib.Path(args.video_path).parent)
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

    # 画面比率
    aspect = W / H

    if W != 1280 or H != 720 or (fps != 30 and fps != 60):
        logger.warning("大きさもしくはfpsが処理対象外のため、再エンコードします: {0}".format(args.video_path))

        # 既存は削除
        if os.path.exists(RESIZE_PATH):
            shutil.rmtree(RESIZE_PATH)

        if os.path.exists(INTERPOLATION_PATH):
            shutil.rmtree(INTERPOLATION_PATH)

        # フォルダ生成
        os.makedirs(RESIZE_PATH)
        os.makedirs(INTERPOLATION_PATH)

        # リサイズpng出力先
        resize_img_path = "{0}/resize{1:012}.png"
        # 補間png出力先
        interpolation_output_path = "{0}/interpolation_{1:012}.png"

        # 縮尺
        scale = width / W

        # オリジナル高さ
        im_height = int(H * scale)

        # 出力ファイルパス
        out_name = '{0}_30fps_{1}.mp4'.format(os.path.basename(args.video_path).replace('.', '_'), "{0:%Y%m%d_%H%M%S}".format(datetime.datetime.now()))
        out_path = os.path.join(base_path, out_name)

        try:
            # 入力ファイル
            cap = cv2.VideoCapture(args.video_path)

            logger.info("元動画読み込み開始", decoration=MLogger.DECORATION_LINE)

            for n in tqdm(range(int(count))):
                # 動画から1枚キャプチャして読み込む
                flag, frame = cap.read()  # Capture frame-by-frame

                # 動画が終わっていたら終了
                if flag == False:
                    break

                # 画像の縦横を指定サイズに変形
                img = Image.fromarray(frame)
                img = img.resize((width, im_height), Image.ANTIALIAS)

                # 黒く塗りつぶす用の背景画像を作成
                bg = Image.new("RGB", [width,height], (0,0,0))

                # 元の画像を、背景画像のセンターに配置
                bg.paste(img, (int((width-img.size[0])/2), int((height-img.size[1])/2)))

                # opencv用に変換
                out_frame = np.asarray(bg)

                # PNG出力
                cv2.imwrite(resize_img_path.format(RESIZE_PATH, n), out_frame)

                # if n % 100 == 0:
                #     logger.info(f"-- 元動画読み込み中 {n}")

            logger.info("元動画読み込み終了", decoration=MLogger.DECORATION_LINE)

            # 補間 --------------------------
            # フレーム補間用比率
            fps_interpolation = fps / 30

            logger.info("補間生成開始", decoration=MLogger.DECORATION_LINE)

            # 最後の１つ手前（補間ができる状態）までループ
            for k in tqdm(range(round(count * (30 / fps)) - 1)):
                # 補間した出力CNT
                inter_cnt = k * fps_interpolation
                # INDEXと比率（整数部と小数部）
                inter_cnt_rate, inter_cnt_idx = math.modf(inter_cnt)
                # logger.debug("フレーム補間: %s -> %s, idx: %s, rate: %s" % ( cnt, inter_cnt, inter_cnt_idx, inter_cnt_rate ))

                # 前のフレーム
                past_frame = cv2.imread(resize_img_path.format(RESIZE_PATH, int(inter_cnt_idx)))
                # 今回のフレーム
                now_frame = cv2.imread(resize_img_path.format(RESIZE_PATH, int(inter_cnt_idx + 1)))

                # 混ぜ合わせる比率
                past_rate = inter_cnt_rate
                now_rate = 1 - inter_cnt_rate

                # フレーム補間をして出力する
                target_output_frame = cv2.addWeighted(past_frame, past_rate, now_frame, now_rate, 0)

                # PNG出力
                cv2.imwrite(interpolation_output_path.format(INTERPOLATION_PATH, k), target_output_frame)

                # if k % 100 == 0:
                #     logger.info(f"-- 補間生成中 {k}")

            # 最後にnowを出力
            cv2.imwrite(interpolation_output_path.format(INTERPOLATION_PATH, k), now_frame)

            logger.info("補間生成終了", decoration=MLogger.DECORATION_LINE)

            logger.info("結合開始", decoration=MLogger.DECORATION_LINE)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, 30.0, (width, height))

            # 結合開始
            for file_path in tqdm(sorted(glob.glob('{0}/interpolation_*.png'.format(INTERPOLATION_PATH)), key=sort_by_numeric)):
                # フレーム
                frame = cv2.imread(file_path)

                # 動画出力
                out.write(frame)

                # if m % 100 == 0:
                #     logger.info(f"-- 結合中 {m}")

            logger.info("結合終了", decoration=MLogger.DECORATION_LINE)

            # 終わったら開放
            out.release()
            cap.release()
        except Exception as e:
            print("再エンコード失敗", e)
            print(traceback.format_exc())

        cv2.destroyAllWindows()

        # 既存は削除
        if os.path.exists(RESIZE_PATH):
            shutil.rmtree(RESIZE_PATH)

        if os.path.exists(INTERPOLATION_PATH):
            shutil.rmtree(INTERPOLATION_PATH)

        logger.info(f'MMD入力用MP4ファイル再生成完了: {out_path}', decoration=MLogger.DECORATION_BOX)
        
        video = cv2.VideoCapture(out_path)
        # 幅
        W = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        # 高さ
        H = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # 総フレーム数
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps
        fps = video.get(cv2.CAP_PROP_FPS)

        logger.info("【再チェック】\n　ファイル名(file name): {0}, ファイルサイズ(file size): {1}\n".format(out_path, os.path.getsize(out_path)) \
                    + "　横(width): {0}, 縦(height): {1}, フレーム数(count of frames): {2}, fps: {3}".format(W, H, count, fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video-path', type=str, dest='video_path', help='video path')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')

    args = parser.parse_args()

    MLogger.initialize(level=args.verbose, is_file=True)

    execute(args)
