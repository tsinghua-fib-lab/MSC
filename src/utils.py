# -*- coding: utf-8 -*-
"""
@author: zgz
"""


def args_printer(args):
    for arg in vars(args):
        print(arg, getattr(args, arg))
