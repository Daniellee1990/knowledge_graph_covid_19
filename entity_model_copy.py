from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
#import coref_ops
import conll
import metrics

class EntityModel:
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = { g:i for i,g in enumerate(config["genres"]) }
    
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None # Load eval data lazily.
    
    def load_lm_embeddings(self, doc_key):
        """ get ELMo embeddings
        """
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_example(self, example, is_training):
        """ for one literature, generate tensors for training
        Args:
            example: dict,
            is_training: boolean
        Returns:
            tokens: (num_sentence * max_sentence_length) tokens[i][j] = word
            context_word_emb: (num_sentence * max_sentence_length * embedding_size), Glove embedding for word
            head_word_emb: (num_sentence * max_sentence_length * embedding_size), Glove embedding for word
            lm_emb: (num_sentence * max_sentence_length * lm_size * lm_layers) ,ELMo embedding
            char_index: (num_sentence * max_sentence_length * max_word_length)
            text_len: [num_sentence], length of every sentence
            speaker_ids: [total_word], every word belong to which speaker
            genre: int, genre index
            is_training: boolean 
            gold_starts: start indices of mentions
            gold_ends, end indices of mentions
            cluster_ids: set
        """
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]
        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]
                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        
        tokens = np.array(tokens)

        speaker_dict = { s:i for i,s in enumerate(set(speakers)) }
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)

        # ELMo embedding
        lm_emb = self.load_lm_embeddings(doc_key)

        example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
            text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)
        
        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors
    
    def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
        text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
            text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids
    
    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
        text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids):
        
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        # embeddings
        # glove embedding + char embedding + elmo embedding?
        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        # get character embedding
        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), \
                char_index) # [num_sentences, max_sentence_length, max_word_length, char_embedding_size]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2), \
                util.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, char_embedding_size]
            flattened_aggregated_char_emb = self.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) 
            # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, \
                util.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
            
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)
    
    def cnn(self, inputs, filter_sizes, num_filters):
        """ concatenate 3 conv1d layer output
        in_channel, out_channel
        """
        num_words = shape(inputs, 0)
        num_chars = shape(inputs, 1)
        input_size = shape(inputs, 2)
        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.variable_scope("conv_{}".format(i)):
                w = tf.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(inputs, w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(h, 1) # [num_words, num_filters]
            outputs.append(pooled)
        
        return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]
