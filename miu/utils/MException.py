# -*- coding: utf-8 -*-
#


class SizingException(Exception):
    def __init__(self, message):
        self.message = message


class MParseException(SizingException):
    def __init__(self, message):
        self.message = message


class MKilledException(SizingException):
    def __init__(self):
        self.message = None

