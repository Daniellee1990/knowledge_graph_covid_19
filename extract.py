import os
import json
import csv
#import nltk
#nltk.download('punkt')
#from nltk import tokenize

def get_stat(entity_types, relations):
    entity_types = list(entity_types)
    #for entity_type in entity_types:
    #    print('entity type: {}'.format(entity_type))

    relations = list(relations)
    #for relation in relations:
    #    print('relation: {}'.format(relation))
    print('total entity type: {}'.format(len(entity_types)))
    print('total relations: {}'.format(len(relations)))

def get_triples(body, entity_types, entity_relations):
    """
    Args:
        body: dict
    Returns:
        entity_types: dict,
        entity_relations: dict
    """

    for entity in body['extractions']:
        entity_name = entity['extraction']['base']
        entity_type = entity['extraction']['type']
        
        type_info = entity_types.get(entity_name, {})
        type_freq = type_info.get(entity_type, 0) + 1
        type_info.update({entity_type: type_freq})
        entity_types.update({entity_name: type_info})
    
    for relation in body['relations']:
        relation_type = relation['kind']
        entity_pair = (relation['elements'][0]['base'], relation['elements'][1]['base'])
        relation_info = entity_relations.get(entity_pair, {})

        relation_freq = relation_info.get(relation_type, 0) + 1
        relation_info.update({relation_type: relation_freq})
        entity_relations.update({entity_pair: relation_info})
    
    return entity_types, entity_relations

def save_types(entity_types, name='entity_types'):
    with open('data/kg-triplet/{}.csv'.format(name), 'w') as csvfile:
        fields = ['subject', 'relation', 'object']
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for entity in entity_types:
            max_type = max(entity_types[entity], key=entity_types[entity].get)
            row = [entity, 'is_type', max_type]
            csvwriter.writerow(row)

def save_relations(entity_relations, name='entity_relations'):
    with open('data/kg-triplet/{}.csv'.format(name), 'w') as csvfile:
        fields = ['subject', 'relation', 'object']
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        for pair in entity_relations:
            max_relation = max(entity_relations[pair], key=entity_relations[pair].get)
            row = [pair[0], max_relation, pair[1]]
            csvwriter.writerow(row)

file_dir = 'data/cord-19_expertsystem_mesh_20200501/arxiv/pdf_json'
entity_types, entity_relations = {}, {}
limit = 2
count = 0

for folder, _, files in os.walk(file_dir):
    for name in files:
        if name.split('.')[-1] == 'json':
            print(name)
            with open('{}/{}'.format(folder, name), 'r') as json_file:
                body = json.load(json_file)
                entity_types, entity_relations = get_triples(body, entity_types, entity_relations)
            
            count += 1
            if count == limit:
                break
    break

save_types(entity_types)
save_relations(entity_relations)

"""
with open('data/0001418189999fea7f7cbe3e82703d71c85a6fe5.json', 'r') as json_file:
    body = json.load(json_file)

    for ele in body['abstract']:
        paragraph = ele['text']
        sentences = tokenize.sent_tokenize(paragraph)
        for sentence in sentences:
            print('sentence: {}'.format(sentence))

    for ele in body['body_text']:
        paragraph = ele['text']
        sentences = tokenize.sent_tokenize(paragraph)
        for sentence in sentences:
            print('sentence: {}'.format(sentence))
"""
