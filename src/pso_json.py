# -*- Mode: Python; tab-width: 8; indent-tabs-mode: nil; python-indent-offset: 4 -*-
# vim:set et sts=4 ts=4 tw=80:
# This Source Code Form is subject to the terms of the MIT License.
# If a copy of the ML was not distributed with this
# file, You can obtain one at https://opensource.org/licenses/MIT

# author: JackRed <jackred@tuta.io>
# Timothée Couble

import json
import os

NAME_DIR = '../config'
EXT = '.json'


def get_increment_name(name, ext):
    i = 0
    while os.path.exists('%s_%d%s' % (name, i, ext)):
        i += 1
    return i


def encode_args(function, name='opso', **kwargs):
    json_opso_args = json.dumps(kwargs, indent=4)
    if not os.path.exists(NAME_DIR):
        os.makedirs(NAME_DIR)
    name = '%s/%s_%s' % (NAME_DIR, function, name)
    name = "%s_%d%s" % (name, get_increment_name(name, EXT), EXT)
    print('config stored in ', name)
    with open(name, 'w') as f:
        f.write(json_opso_args)


def decode_args(function, name='opso', n=0):
    name = "%s/%s_%s_%d%s" % (NAME_DIR, function, name, n, EXT)
    with open(name, 'r') as f:
        args = f.read()
    return json.loads(args)


def get_boundary_config(n=0, filename='opso_boundary'):
    name = "%s/%s_%d%s" % (NAME_DIR, filename, n, EXT)
    with open(name, 'r') as f:
        args = f.read()
    return json.loads(args)
