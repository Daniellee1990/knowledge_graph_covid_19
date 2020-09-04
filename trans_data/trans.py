#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

name = 'dev'

def add_speaker(name):
    new_list = []
    with open('data/json/{}.json'.format(name), 'r') as f:
        for line in f.readlines():
            body = json.loads(line)
            speakers = []
            speaker_id = 1
            for sentence in body['sentences']:
                sentence_speaker = [speaker_id for _ in sentence]
                speakers.append(sentence_speaker)
                speaker_id += 1
            body.update({
                'speakers': speakers
            })
            new_list.append(body)

    with open('data/json/{}_new.json'.format(name), 'w') as f:
        for body in new_list:
            f.write('{}\n'.format(json.dumps(body)))
    
if __name__ == '__main__':
    names = ['dev', 'test', 'train']
    for name in names:
        add_speaker(name)
