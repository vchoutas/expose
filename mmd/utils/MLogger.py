# -*- coding: utf-8 -*-
#
from datetime import datetime
import logging
import traceback
import threading
import sys
import os
import json
import locale

from mmd.utils.MException import MKilledException


class MLogger():

    DECORATION_IN_BOX = "in_box"
    DECORATION_BOX = "box"
    DECORATION_LINE = "line"
    DEFAULT_FORMAT = "%(message)s [%(funcName)s][P-%(process)s](%(asctime)s)"

    DEBUG_FULL = 2
    TEST = 5
    TIMER = 12
    FULL = 15
    INFO_DEBUG = 22
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    # 翻訳モード
    # 読み取り専用：翻訳リストにない文字列は入力文字列をそのまま出力する
    MODE_READONLY = 0
    # 更新あり：翻訳リストにない文字列は出力する
    MODE_UPDATE = 1

    # 翻訳モード
    mode = MODE_READONLY
    # 翻訳言語優先順位
    langs = ["en_US", "ja_JP"]
    # 出力対象言語
    target_lang = "ja_JP"
    # システム全体のロギングレベル
    total_level = logging.INFO
    # システム全体の開始出力日時
    outout_datetime = ""

    # デフォルトログファイルパス
    default_out_path = ""

    messages = {}
    logger = None

    def __init__(self, module_name, level=logging.INFO, out_path=None):
        self.module_name = module_name
        self.default_level = level

        # ロガー
        self.logger = logging.getLogger("expose_mmd").getChild(self.module_name)

        if not out_path:
            # クラス単位の出力パスがない場合、デフォルトパス
            out_path = self.default_out_path

        # # 一旦既存のファイルハンドラは削除
        # for f in self.logger.handlers:
        #     if isinstance(f, logging.FileHandler):
        #         self.logger.removeHandler(f)

        if out_path:
            # ファイル出力ハンドラ
            fh = logging.FileHandler(out_path)
            fh.setLevel(self.default_level)
            fh.setFormatter(logging.Formatter(self.DEFAULT_FORMAT))
            self.logger.addHandler(fh)
    
    def time(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.TIMER
        self.print_logger(msg, *args, **kwargs)

    def info_debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.INFO_DEBUG
        self.print_logger(msg, *args, **kwargs)

    def test(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.TEST
        self.print_logger(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.DEBUG
        self.print_logger(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.INFO
        self.print_logger(msg, *args, **kwargs)

    # ログレベルカウント
    def count(self, msg, fno, fnos, *args, **kwargs):
        last_fno = 0

        if fnos and len(fnos) > 0 and fnos[-1] > 0:
            last_fno = fnos[-1]
        
        if not fnos and kwargs and "last_fno" in kwargs and kwargs["last_fno"] > 0:
            last_fno = kwargs["last_fno"]

        if last_fno > 0:
            if not kwargs:
                kwargs = {}
                
            kwargs["level"] = logging.INFO
            log_msg = "-- {0}フレーム目:終了({1}％){2}".format(fno, round((fno / last_fno) * 100, 3), msg)
            self.print_logger(log_msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.WARNING
        self.print_logger(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.ERROR
        self.print_logger(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.CRITICAL
        self.print_logger(msg, *args, **kwargs)

    # 実際に出力する実態
    def print_logger(self, msg, *args, **kwargs):

        if "is_killed" in threading.current_thread()._kwargs and threading.current_thread()._kwargs["is_killed"]:
            # 停止命令が出ている場合、エラー
            raise MKilledException()

        target_level = kwargs.pop("level", logging.INFO)
        if self.total_level <= target_level and self.default_level <= target_level:
            # システム全体のロギングレベルもクラス単位のロギングレベルもクリアしてる場合のみ出力

            # モジュール名を出力するよう追加
            extra_args = {}
            extra_args["module_name"] = self.module_name

            trans_msg = msg
            if msg in self.messages:
                # メッセージがある場合、それで出力する
                trans_msg = self.messages[msg]

            if self.mode == MLogger.MODE_UPDATE:
                # 更新モードである場合、辞書に追記
                for lang in self.langs:
                    messages_path = os.path.join("locale", lang, "messages.json")
                    try:
                        with open(messages_path, 'r', encoding="utf-8") as f:
                            msgs = json.load(f)

                            if msg not in msgs:
                                # ない場合、追加(オリジナル言語の場合、そのまま。違う場合は空欄)
                                msgs[msg] = msg if self.target_lang == lang else ""

                            with open(messages_path, 'w', encoding="utf-8") as f:
                                json.dump(msgs, f, ensure_ascii=False)
                    except Exception as e:
                        print("*** Message Update ERROR ***\n%s", traceback.format_exc())

            # ログレコード生成
            if args and isinstance(args[0], Exception) or (args and len(args) > 1 and isinstance(args[0], Exception)):
                trans_msg = f"{trans_msg}\n\n{traceback.format_exc()}"
                args = None
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, args, None, None, self.module_name)
            else:
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, trans_msg, args, None, self.module_name)
            
            target_decoration = kwargs.pop("decoration", None)
            title = kwargs.pop("title", None)

            print_msg = str(trans_msg)
            if args:
                print_msg = print_msg.format(*args)

            if target_decoration:
                if target_decoration == MLogger.DECORATION_BOX:
                    output_msg = self.create_box_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_LINE:
                    output_msg = self.create_line_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_IN_BOX:
                    output_msg = self.create_in_box_message(print_msg, target_level, title)
                else:
                    output_msg = self.create_simple_message(print_msg, target_level, title)
            else:
                output_msg = self.create_simple_message(print_msg, target_level, title)
        
            # 出力
            try:
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, output_msg, None, None, self.module_name)
                self.logger.handle(log_record)
            except Exception as e:
                raise e
            
    def create_box_message(self, msg, level, title=None):
        msg_block = []
        msg_block.append("■■■■■■■■■■■■■■■■■")

        if level == logging.CRITICAL:
            msg_block.append("■　**CRITICAL**　")

        if level == logging.ERROR:
            msg_block.append("■　**ERROR**　")

        if level == logging.WARNING:
            msg_block.append("■　**WARNING**　")

        if level <= logging.INFO and title:
            msg_block.append("■　**{0}**　".format(title))

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        msg_block.append("■■■■■■■■■■■■■■■■■")

        return "\n".join(msg_block)

    def create_line_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("-- {0} --------------------".format(msg_line))

        return "\n".join(msg_block)

    def create_in_box_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        return "\n".join(msg_block)

    def create_simple_message(self, msg, level, title=None):
        msg_block = []
        
        for msg_line in msg.split("\n"):
            # msg_block.append("[{0}] {1}".format(logging.getLevelName(level)[0], msg_line))
            msg_block.append(msg_line)
        
        return "\n".join(msg_block)
    
    @classmethod
    def initialize(cls, langs=["en_US", "ja_JP"], mode=MODE_READONLY, level=logging.INFO, out_path=None):
        # logging.basicConfig(level=level)
        logging.basicConfig(level=level, format=cls.DEFAULT_FORMAT)
        cls.langs = langs
        cls.total_level = level
        cls.mode = mode

        outout_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.now())
        cls.outout_datetime = outout_datetime

        # ファイル出力ありの場合、ログファイル名生成
        if not out_path:
            os.makedirs("log", exist_ok=True)
            cls.default_out_path = "log/expose_mmd_{0}.log".format(outout_datetime)
        else:
            cls.default_out_path = out_path
        
        if mode == MLogger.MODE_UPDATE:
            # 更新版の場合、必要なディレクトリを全部作成する
            for lang in langs:
                os.makedirs(os.path.join("locale", lang), exist_ok=True)
                messages_path = os.path.join("locale", lang, "messages.json")
                if not os.path.exists(messages_path):
                    try:
                        with open(messages_path, 'w', encoding="utf-8") as f:
                            json.dump({}, f, ensure_ascii=False)
                    except Exception as e:
                        print("*** Message Dump ERROR ***\n%s", traceback.format_exc())

        # 実行環境に応じたローカル言語
        lang = locale.getdefaultlocale()[0]
        if lang not in langs:
            # 実行環境言語に対応した言語が出力対象外である場合、第一言語を出力する
            cls.target_lang = langs[0]
        else:
            # 実行環境言語に対応した言語が出力対象である場合、その言語を出力する
            cls.target_lang = "ja_JP"
            # cls.target_lang = lang

        # メッセージファイルパス
        try:
            with open(os.path.join("locale", cls.target_lang, "messages.json"), 'r', encoding="utf-8") as f:
                cls.messages = json.load(f)
        except Exception as e:
            print("*** Message Load ERROR ***\n%s", traceback.format_exc())


