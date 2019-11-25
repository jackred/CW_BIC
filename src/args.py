# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

import argparse

# TODO: config file for boundary


def parse_args(name):
    argp = argparse.ArgumentParser(name)
    argp.add_argument('-f', '--function', dest='function',
                      help='function to optimize',
                      choices=['linear', 'cubic', 'tanh', 'sine',
                               'complex', 'xor'])
    return argp


def opso_args():
    argp = parse_args('OPSO training PSO to optimize ANN')
    argp.add_argument('-onc', '--opso-number-config', dest='onc',
                      default=0, type=int,
                      help='number of the config file for opso')
    argp.add_argument('-obc', '--opso-boundary-config', dest='obc',
                      default=0, type=int,
                      help='number of the config file for opso')
    return argp


def pso_args():
    argp = parse_args('PSO to optimize ANN')
    argp.add_argument('-pnc', '--pso-number-config', dest='pnc',
                      default=0, type=int,
                      help='number of the config file for pso')
    return argp


def ann_args():
    argp = parse_args('ANN to approximate math functions')
    argp.add_argument('-anc', '--ann-number-config', dest='anc',
                      default=0, type=int,
                      help='number of the config file for ann')
    return argp
