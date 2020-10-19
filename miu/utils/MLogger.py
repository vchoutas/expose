# -*- coding: utf-8 -*-
#
from datetime import datetime
import logging
import traceback
import threading
import os

from miu.utils.MException import MKilledException


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
    
    total_level = logging.INFO
    is_file = False
    outout_datetime = ""
    
    logger = None

    def __init__(self, module_name, level=logging.INFO):
        self.module_name = module_name
        self.default_level = level
        self.child = False

        # ロガー
        self.logger = logging.getLogger("VmdOutput").getChild(self.module_name)

        for f in self.logger.handlers:
            if isinstance(f, logging.FileHandler):
                # 既存のファイルハンドラはすべて削除
                self.logger.removeHandler(f)

        # ファイル出力ありの場合、ハンドラ紐付け
        if not os.path.exists("log"):
            os.makedirs("log", exist_ok=True)

        # ファイル出力ハンドラ
        fh = logging.FileHandler("log/AutoTrace{0}.log".format(self.outout_datetime))
        fh.setLevel(self.default_level)
        fh.setFormatter(logging.Formatter(self.DEFAULT_FORMAT))
        self.logger.addHandler(fh)

    def copy(self, options):
        self.is_file = options.is_file
        self.outout_datetime = options.outout_datetime
        self.monitor = options.monitor
        self.child = True

        for f in self.logger.handlers:
            if isinstance(f, logging.StreamHandler):
                f.setStream(options.monitor)

    def time(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.TIMER
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)

    def info_debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = self.INFO_DEBUG
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)

    def test(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.TEST
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)
    
    def debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.DEBUG
        kwargs["time"] = True
        self.print_logger(msg, *args, **kwargs)
    
    def info(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}
            
        kwargs["level"] = logging.INFO
        self.print_logger(msg, *args, **kwargs)

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
        # if self.logger.isEnabledFor(target_level) and self.default_level <= target_level:
        if self.total_level <= target_level and self.default_level <= target_level:
            # モジュール名を出力するよう追加
            extra_args = {}
            extra_args["module_name"] = self.module_name

            # ログレコード生成
            if args and isinstance(args[0], Exception):
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, "{0}\n\n{1}".format(msg, traceback.format_exc()), None, None, self.module_name)
            else:
                log_record = self.logger.makeRecord('name', target_level, "(unknown file)", 0, msg, args, None, self.module_name)
            
            target_decoration = kwargs.pop("decoration", None)
            title = kwargs.pop("title", None)
            is_time = kwargs.pop("time", None)

            if is_time:
                # 時間表記が必要な場合、表記追加
                print_msg = "{message} [{funcName}]({now:%H:%M:%S.%f})".format(message=log_record.getMessage(), funcName=self.module_name, now=datetime.now())
            else:
                print_msg = "{message}".format(message=log_record.getMessage())
            
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
            msg_block.append("■■ {0} --------------------".format(msg_line))

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
    def initialize(cls, level=logging.INFO, is_file=False):
        # logging.basicConfig(level=level)
        logging.basicConfig(level=level, format=cls.DEFAULT_FORMAT)
        cls.total_level = level
        cls.is_file = is_file
        cls.outout_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.now())
