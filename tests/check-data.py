import os
import csv
import sys
import json
import nltk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def check_entity(name):
    # handle entity
    entity_types = set()
    with open('../../data-center/KG/entity/pmcid/{}.txt'.format(name), 'r') as f:
        body = json.load(f)
        for e in body['passages']:
            text = e['text']
            sent_text = nltk.sent_tokenize(text)
            print('text sentence length: {}'.format(len(sent_text)))
            for sent in sent_text:
                word_tokenize = nltk.word_tokenize(sent)
                print('sentence: {}'.format(word_tokenize))

def check_relation():
    for root, dirs, files in os.walk('../../data-center/KG/entity/pmid_abs'):
        for name in files:
            path = os.path.join(root, name)
            print('checking {}'.format(name))
            with open(path, 'r') as f:
                body = json.load(f)
                passages = body['passages']
                for psg in passages:
                    if len(psg['relations']) > 0:
                        print('test relations')
                        print(psg['relations'])

def list_relation(name):
    samples = []
    with open('../../data-center/KG/{}.csv'.format(name), 'r') as f:
        body = csv.reader(f, delimiter='\t')
        cnt = 0
        for row in body:
            samples.append(row)
            cnt += 1
            if cnt == 10000:
                break

    with open('../../data-center/KG/{}_sample.csv'.format(name), 'w') as f:
        csvwriter = csv.writer(f)
        for row in samples:
            csvwriter.writerow(row)

    print('Done')

if __name__ == '__main__':
    # name = 'entity/pmcid/pubtator_67173_PMC2180621.json'
    name = 'chemicals_diseases_relation'

    #check_relation()
    list_relation(name)
    


