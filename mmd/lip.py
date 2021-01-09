# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import pathlib
import _pickle as cPickle

# import vision essentials
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from spleeter.separator import Separator
from spleeter.audio.adapter import get_default_audio_adapter
from mmd.utils.MLogger import MLogger

logger = MLogger(__name__)

def execute(args):
    try:
        logger.info('音声認識処理開始: {0}', args.audio_file, decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.audio_file):
            logger.error("指定された音声ファイルパスが存在しません。\n{0}", args.video_file, decoration=MLogger.DECORATION_BOX)
            return False, None

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.audio_file).parent) if not args.parent_dir else args.parent_dir

        audio_adapter = get_default_audio_adapter()
        sample_rate = 44100
        waveform, _ = audio_adapter.load(args.audio_file, sample_rate=sample_rate)

        # 音声と曲に分離
        separator = Separator('spleeter:2stems')

        # Perform the separation :
        prediction = separator.separate(waveform)

        # 音声データ
        vocals = prediction['vocals']

        audio_adapter.save(f"{base_path}/vocals.wav", vocals, separator._sample_rate, "wav", "16k")



        logger.info('音声認識処理終了: {0}', base_path, decoration=MLogger.DECORATION_BOX)

        return True
    except Exception as e:
        logger.critical("音声認識で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
 

 