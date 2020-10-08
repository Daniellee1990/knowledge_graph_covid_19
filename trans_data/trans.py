#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def flatten(l):
    return [item for sublist in l for item in sublist]

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

def get_labels(names=['dev', 'test', 'train']):
    entity_map = {}
    relation_map = {}
    for name in names:
        with open('data/json/{}_raw.json'.format(name), 'r') as f:
            for line in f.readlines():
                body = json.loads(line)
                ner = body['ner']
                relations = body['relations']
                
                for sentence_entity, sentence_relation in zip(ner, relations):
                    for e in sentence_entity:
                        entity = e[2]
                        if entity not in entity_map:
                            label = len(entity_map) + 1
                            entity_map.update({
                                entity: label
                            })
                    for r in sentence_relation:
                        relation = r[4]
                        if relation not in relation_map:
                            label = len(relation_map) + 1
                            relation_map.update({
                                relation: label
                            })

    return entity_map, relation_map

def flatten_entity_relation(name):
    with open('data/json/{}_raw.json'.format(name), 'r') as f:
        new_list = []
        for line in f.readlines():
            body = json.loads(line)
            entity_starts, entity_ends, entity_labels = [], [], []
            relation1_starts, relation1_ends, relation2_starts, relation2_ends, relation_labels = \
                [], [], [], [], []
            
            for sentence_entity in body['ner']:
                for e in sentence_entity:
                    entity_starts.append(e[0])
                    entity_ends.append(e[1])
                    entity_labels.append(entity_map[e[2]])

            for sentence_relation in body['relations']:
                for r in sentence_relation:
                    relation1_starts.append(r[0])
                    relation1_ends.append(r[1])
                    relation2_starts.append(r[2])
                    relation2_ends.append(r[3])
                    relation_labels.append(relation_map[r[4]])
            
            body.update({
                'entity_starts': entity_starts,
                'entity_ends': entity_ends,
                'entity_labels': entity_labels,
                'relation1_starts': relation1_starts,
                'relation1_ends': relation1_ends,
                'relation2_starts': relation2_starts,
                'relation2_ends': relation2_ends,
                'relation_labels': relation_labels
            })

            new_list.append(copy.deepcopy(body))

    with open('data/json/{}_new.json'.format(name), 'w') as f:
        for body in new_list:
            f.write('{}\n'.format(json.dumps(body)))        

if __name__ == '__main__':
    names = ['dev', 'test', 'train']
    # add speaker to sentence
    #for name in names:
    #    add_speaker(name)

    entity_map, relation_map = get_labels(names)
    for name in names:
        flatten_entity_relation(name)


