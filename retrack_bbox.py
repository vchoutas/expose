# -*- coding: utf-8 -*-
import os
import argparse
import glob
import sys
import json
import csv
import pathlib

# import vision essentials
import numpy as np
from tqdm import tqdm

from miu.utils.MLogger import MLogger
from miu.utils.MServiceUtils import sort_by_numeric
import miu.config as mconfig


logger = MLogger(__name__, level=MLogger.DEBUG)

def execute(args):
    logger.info(f'人物再追跡処理開始: {args.process_dir}', decoration=MLogger.DECORATION_BOX)

    if not os.path.exists(args.process_dir):
        logger.error("指定された処理用ディレクトリパスが存在しません。: {0}".format(args.process_dir))
        return

    if not os.path.exists(args.order_file_path):
        logger.error("指定された順番指定用ファイルパスが存在しません。: {0}".format(args.order_file_path))
        return

    with open(args.order_file_path, "r") as f:
        reader = csv.reader(f)
        for ridx, rows in enumerate(reader):
            # 1行1ユーザー
            bbox_json_path = os.path.join(args.process_dir, "**", "bbox_*.json")
            for json_path in tqdm(glob.glob(bbox_json_path)):
                bbox_dict = {}
                with open(json_path, 'r') as f:
                    bbox_dict = json.load(f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--process-dir', type=str, dest='process_dir', help='process dir path')
    parser.add_argument('--order-file-path', type=str, dest='order_file_path', help='specifying tracking')
    parser.add_argument('--verbose', type=int, dest='verbose', default=20, help='log level')

    args = parser.parse_args()

    MLogger.initialize(level=args.verbose, is_file=True)

    execute(args)

