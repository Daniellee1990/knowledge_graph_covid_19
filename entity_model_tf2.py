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

import util_tf2
import coref_ops
import conll
import metrics
from entity_eval import EntityEvaluator

from tensorflow.python.keras.backend import set_session
tf.compat.v1.disable_eager_execution()

class EntityModel:
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util_tf2.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util_tf2.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util_tf2.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = { g:i for i,g in enumerate(config["genres"]) }

        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None # Load eval data lazily.

    def _get_input_structure(self):
        input_props = []
        input_props.append((tf.string, [None, None])) # Tokens.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size])) # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size])) # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers])) # LM embeddings.
        input_props.append((tf.int32, [None, None, None])) # Character indices.
        input_props.append((tf.int32, [None])) # Text lengths.
        input_props.append((tf.int32, [None])) # Speaker IDs.
        input_props.append((tf.int32, [])) # Genre.
        input_props.append((tf.bool, [])) # Is training.
        input_props.append((tf.int32, [None])) # entity_starts
        input_props.append((tf.int32, [None])) # entity_ends
        input_props.append((tf.int32, [None])) # entity_labels
        input_props.append((tf.int32, [None, 2])) # relation_starts,
        input_props.append((tf.int32, [None, 2])) # relation_ends,
        input_props.append((tf.int32, [None])) # relation_labels,
        input_props.append((tf.int32, [None])) # Gold starts.
        input_props.append((tf.int32, [None])) # Gold ends.
        input_props.append((tf.int32, [None])) # Cluster ids.

        return input_props

    def start(self, session):
        input_props = self._get_input_structure()
        
        self.queue_input_tensors = [tf.compat.v1.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.queue.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)

        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        train_examples = self.read_data()
        enqueue_thread = threading.Thread(target=self.enqueue_loop, args=(train_examples, session))
        # enqueue_thread.daemon = True
        enqueue_thread.start()
    
    def train(self):
        # training process
        self.predictions, self.loss = self.get_predictions_and_loss(*self.input_tensors)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        #'''
        self.reset_global_step = tf.compat.v1.assign(self.global_step, 0)
        learning_rate = tf.compat.v1.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                self.config["decay_frequency"], self.config["decay_rate"], staircase=True)
        trainable_params = tf.compat.v1.trainable_variables()
        gradients = tf.gradients(ys=self.loss, xs=trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam" : tf.compat.v1.train.AdamOptimizer,
            "sgd" : tf.compat.v1.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)
        #'''

    def read_data(self):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]
        return train_examples
    
    def enqueue_loop(self, train_examples, session):
        while True:
            random.shuffle(train_examples)
            for example in train_examples:
                print('put in queue')
                tensorized_example = self.tensorize_example(example, is_training=True)
                feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                session.run(self.enqueue_op, feed_dict=feed_dict)

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
            entity_starts: [num_entity], start indices of entities
            entity_ends: [num_entity], end indices of entities
            entity_labels: [num_entity]
            relation_starts: [num_relation, 2]
            relation_ends: [num_relation, 2]
            relation_labels: [num_relation]
            gold_starts: start indices of mentions
            gold_ends, end indices of mentions
            cluster_ids: set
        """
        clusters = example["clusters"]

        gold_mentions = sorted(tuple(m) for m in util_tf2.flatten(clusters))
        gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util_tf2.flatten(example["speakers"])

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
        genre = self.genres.get(doc_key[:2], len(self.genres) - 1)
        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        
        entity_starts = np.array(example['entity_starts'])
        entity_ends = np.array(example['entity_ends'])
        entity_labels = np.array(example['entity_labels'])
        
        relation_starts = example['relation_starts']
        if not len(relation_starts): 
            relation_starts = np.empty(shape=(0,2))
        relation_starts = np.array(relation_starts)
        relation_ends = example['relation_ends']
        if not len(relation_ends): 
            relation_ends = np.empty(shape=(0,2))
        relation_ends = np.array(relation_ends)
        
        relation_labels = np.array(example['relation_labels'])

        # ELMo embedding
        lm_emb = self.load_lm_embeddings(doc_key)

        example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
            text_len, speaker_ids, genre, is_training, entity_starts, entity_ends, entity_labels, \
                relation_starts, relation_ends, relation_labels, gold_starts, gold_ends, cluster_ids)
        
        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors
    
    def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
        text_len, speaker_ids, genre, is_training, entity_starts, entity_ends, entity_labels, \
            relation_starts, relation_ends, relation_labels, gold_starts, gold_ends, cluster_ids):
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
            text_len, speaker_ids, genre, is_training, entity_starts, entity_ends, entity_labels, \
                relation_starts, relation_ends, relation_labels, gold_starts, gold_ends, cluster_ids
    
    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, \
        text_len, speaker_ids, genre, is_training, entity_starts, entity_ends, entity_labels, \
            relation_starts, relation_ends, relation_labels, gold_starts, gold_ends, cluster_ids):
        
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(input=context_word_emb)[0]
        max_sentence_length = tf.shape(input=context_word_emb)[1]

        # embeddings
        # glove embedding + char embedding + elmo embedding
        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        # character embedding
        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(tf.compat.v1.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]), \
                char_index) # [num_sentences, max_sentence_length, max_word_length, char_embedding_size]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util_tf2.shape(char_emb, 2), \
                util_tf2.shape(char_emb, 3)]) # [num_sentences * max_sentence_length, max_word_length, char_embedding_size]
            flattened_aggregated_char_emb = self.cnn(flattened_char_emb, self.config["filter_widths"], self.config["filter_size"]) 
            # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length, \
                util_tf2.shape(flattened_aggregated_char_emb, 1)]) # [num_sentences, max_sentence_length, emb]
            
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)
        
        # ELMo embedding
        # lm_emb: [num_sentence, max_sentence_length, lm_size, lm_layers]
        lm_emb_size = util_tf2.shape(lm_emb, 2)
        lm_num_layers = util_tf2.shape(lm_emb, 3)
        with tf.compat.v1.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(tf.compat.v1.get_variable("lm_scores", [lm_num_layers], initializer=tf.compat.v1.constant_initializer(0.0)))
            self.lm_scaling = tf.compat.v1.get_variable("lm_scaling", [], initializer=tf.compat.v1.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        # [num_sentences * max_sentence_length * lm_emb_size, lm_emb_layers]
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights, 1)) 
        # [num_sentences * max_sentence_length * emb, 1]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)

        # concatenate embeddings
        context_emb = tf.concat(context_emb_list, 2) # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2) # [num_sentences, max_sentence_length, emb]
        # dropout
        context_emb = tf.nn.dropout(context_emb, 1 - (self.lexical_dropout)) # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, 1 - (self.lexical_dropout)) # [num_sentences, max_sentence_length, emb]

        # embedding part done

        # sequence_mask:
        # orignal tensor t[d_1, d_2,..., d_n]
        # mask[i_1, i_2, ..., i_n, j] = t[i_1, i_2, ..., i_n] < j
        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length) # [num_sentence, max_sentence_length]

        # bi-directional lstm
        # every word gets an embedding
        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask) # [num_words, emb]
        num_words = util_tf2.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.compat.v1.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]), genre) # [emb]

        # handle spans
        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1), [1, max_sentence_length]) # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask) # [num_words]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask) # [num_words]

        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1), [1, self.max_span_width]) 
        # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width), 0) 
        # [num_words, max_span_width]
        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices, candidate_starts) 
        # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends, num_words - 1)) 
        # [num_words, max_span_width]
        
        # candidate spans must come from the same sentence
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices, candidate_end_sentence_indices)) 
        # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1]) # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tensor=tf.reshape(candidate_starts, [-1]), mask=flattened_candidate_mask) # [num_candidates]
        candidate_ends = tf.boolean_mask(tensor=tf.reshape(candidate_ends, [-1]), mask=flattened_candidate_mask) # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tensor=tf.reshape(candidate_start_sentence_indices, [-1]), mask=flattened_candidate_mask) 
        # [num_candidates]

        # get labels
        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)
        # [num_candidates]
        candidate_entity_labels = self.get_entity_labels(candidate_starts, candidate_ends, entity_starts, entity_ends, entity_labels)
        # [num_candidates]

        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts, candidate_ends) # [num_candidates, emb]
        candidate_mention_scores =  self.get_mention_scores(candidate_span_emb) # [num_candidates, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1) # [num_candidates]

        # filter out part of spans
        k = tf.cast(tf.floor(tf.cast(tf.shape(input=context_outputs)[0], dtype=tf.float32) * self.config["top_span_ratio"]), dtype=tf.int32)
        top_span_indices = coref_ops.extract_spans(
            tf.expand_dims(candidate_mention_scores, 0),
            tf.expand_dims(candidate_starts, 0),
            tf.expand_dims(candidate_ends, 0),
            tf.expand_dims(k, 0),
            util_tf2.shape(context_outputs, 0),
            True) # [1, k]
        
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0) # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices) # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices) # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices) # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices) # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices) # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices) # [k]
        top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts) # [k]
        
        top_span_entity_labels = tf.gather(candidate_entity_labels, top_span_indices) # [k]

        # check antecedents
        c = tf.minimum(self.config["max_top_antecedents"], k)
        # c: number of antecedents

        # print('test config', self.config['coarse_to_fine'])
        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
                self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = \
                self.distance_pruning(top_span_emb, top_span_mention_scores, c)

        # relation loss function
        relation_scores = self.get_relation_scores(top_span_emb)

        # co-reference loss function
        '''
        dummy_scores = tf.zeros([k, 1]) # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.compat.v1.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents) # [k, c, emb]
                top_slow_antecedent_scores = self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, \
                        top_antecedent_offsets, top_span_speaker_ids, genre_emb) # [k, c]
                top_antecedent_scores = top_fast_antecedent_scores + top_slow_antecedent_scores # [k, c]
                
                top_antecedent_weights = tf.nn.softmax(tf.concat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb], 1) # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(input_tensor=tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb, axis=1) # [k, emb]
                with tf.compat.v1.variable_scope("f"):
                    f = tf.sigmoid(util_tf2.projection(tf.concat([top_span_emb, attended_span_emb], 1), \
                        util_tf2.shape(top_span_emb, -1))) # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]
        # get candidates label
        self.top_antecedent_labels = self.get_antecedent_labels(top_span_cluster_ids, top_antecedents, top_antecedents_mask)
        '''

        #loss = self.softmax_loss(top_antecedent_scores, self.top_antecedent_labels) # [k]
        #loss = tf.reduce_sum(input_tensor=loss) # []

        # entity scores
        entity_scores = self.get_entity_scores(top_span_emb)
        entity_labels_mask = self.get_entity_label_mask(top_span_entity_labels)

        # entity loss function
        loss = self.entity_loss(entity_scores, entity_labels_mask) # [k]
        loss = tf.reduce_sum(input_tensor=loss) # []

        #return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, \
        #    top_antecedents, top_antecedent_scores], loss
        return [entity_scores, entity_labels_mask], loss

    def get_antecedent_labels(self, top_span_cluster_ids, top_antecedents, top_antecedents_mask):
        """
        Args:
            top_span_cluster_ids: [k], cluster of spans
            top_antecedents: [k, c], index of antecedents for every span
            top_antecedents_mask: [k, c], pair validation indicator
        Returns:
            top_antecedent_labels: [k, c+1]
        """
        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents) # [k, c]
        top_antecedent_cluster_ids += tf.cast(tf.math.log(tf.cast(top_antecedents_mask, dtype=tf.float32)), dtype=tf.int32) # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1)) # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1) # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(input_tensor=pairwise_labels, axis=1, keepdims=True)) # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1) # [k, c + 1]

        return top_antecedent_labels

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        """ bi-directional lstm nn
        Args:
            text_emb: [num_sentences, max_sentence_length, emb]
            text_len: [num_sentences], length of every sentence
            text_len_mask: [num_sentence, max_sentence_length]
        Returns:

        """
        num_sentences = tf.shape(input=text_emb)[0]
        current_inputs = text_emb # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.compat.v1.variable_scope("layer_{}".format(layer)):
                with tf.compat.v1.variable_scope("fw_cell"):
                    cell_fw = util_tf2.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
                with tf.compat.v1.variable_scope("bw_cell"):
                    cell_bw = util_tf2.CustomLSTMCell(self.config["contextualization_size"], num_sentences, self.lstm_dropout)
                #state_fw = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), \
                #    tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                #state_bw = tf.nn.rnn_cell.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), \
                #    tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))
                state_fw = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]), \
                    tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]), \
                    tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw
                )

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2) # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, 1 - (self.lstm_dropout))
                if layer > 0:
                    highway_gates = tf.sigmoid(util_tf2.projection(text_outputs, util_tf2.shape(text_outputs, 2))) 
                    # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)
    
    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(input=emb)[0]
        max_sentence_length = tf.shape(input=emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank  == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util_tf2.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        
        return tf.boolean_mask(tensor=flattened_emb, mask=tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1), tf.expand_dims(candidate_starts, 0)) 
        # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1), tf.expand_dims(candidate_ends, 0)) 
        # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end) 
        # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.cast(same_span, dtype=tf.int32)) 
        # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0) 
        # [num_candidates]
        return candidate_labels
    
    def get_entity_labels(self, candidate_starts, candidate_ends, entity_starts, entity_ends, entity_labels):
        same_start = tf.equal(tf.expand_dims(entity_starts, 1), tf.expand_dims(candidate_starts, 0))
        same_end = tf.equal(tf.expand_dims(entity_ends, 1), tf.expand_dims(candidate_ends, 0))
        same_span = tf.logical_and(same_start, same_end)
        candidate_entity_labels = tf.matmul(tf.expand_dims(entity_labels, 0), tf.cast(same_span, dtype=tf.int32))
        candidate_entity_labels = tf.squeeze(candidate_entity_labels, 0)
        return candidate_entity_labels
    
    def get_relation_labels(self, candidate_starts, candidate_ends, relation_starts, relation_ends, relation_labels):
        """
        Args:
            candidate_starts: [k],
            candidate_ends: [k],
            relation1_starts: [l],
            relation1_ends: [l],
            relation2_starts: [l],
            relation2_ends: [l],
            relation_labels: [l]
        Returns:
            relation_labels: [k, k]
        """
        pass

    def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        """
        Args:
            head_emb: [num_words, emb],
            context_outputs: [num_words, emb],
            span_starts: [num_candidates], word index
            span_ends: [num_candidates], word index
        Returns:
            span_emb: [num_candidates, emb], span embedding
        """
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts) # [num_candidates, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends) # [num_candidates, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts # [num_candidates], with of every span

        if self.config["use_features"]:
            # span length feature
            span_width_index = span_width - 1 # [num_candidates]
            span_width_emb = tf.gather(tf.compat.v1.get_variable("span_width_embeddings", \
                [self.config["max_span_width"], self.config["feature_size"]]), span_width_index) # [num_candidates, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, 1 - (self.dropout))
            span_emb_list.append(span_width_emb)

        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + \
                tf.expand_dims(span_starts, 1) # [num_candidates, max_span_width]
            span_indices = tf.minimum(util_tf2.shape(context_outputs, 0) - 1, span_indices) # [num_candidates, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices) # [num_candidates, max_span_width, emb]
            with tf.compat.v1.variable_scope("head_scores"):
                self.head_scores = util_tf2.projection(context_outputs, 1) # [num_words, 1]
            span_head_scores = tf.gather(self.head_scores, span_indices) # [num_candidates, max_span_width, 1]
            span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) 
            # [num_candidates, max_span_width, 1]
            span_head_scores += tf.math.log(span_mask) # [num_candidates, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1) # [num_candidates, max_span_width, 1]
            span_head_emb = tf.reduce_sum(input_tensor=span_attention * span_text_emb, axis=1) # [num_candidates, emb]
            span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1) # [num_candidates, emb]
        return span_emb # [num_candidates, emb]
    
    def get_mention_scores(self, span_emb):
        """
        Args:
            span_emb: [num_candidates, emb]
        Returns:
            span_mention_scores: [num_candidates, 1], span mention score
        """
        with tf.compat.v1.variable_scope("mention_scores"):
            return util_tf2.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, 1]

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.cast(is_training, dtype=tf.float32) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        """
        Args:
            top_span_emb: [k, emb]
            top_span_mention_scores: [k], mention scores
            c: top number
        """
        k = util_tf2.shape(top_span_emb, 0)
        top_span_range = tf.range(k) # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0) # [k, k]
        antecedents_mask = antecedent_offsets >= 1 # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores, 0) # [k, k]
        fast_antecedent_scores += tf.math.log(tf.cast(antecedents_mask, dtype=tf.float32)) # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb) # [k, k]

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False) # [k, c]
        top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents) # [k, c]
        top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
        top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
        """
        Args:
            top_span_emb: [k, emb],
            top_span_mention_scores: [k]
            c: top number
        """
        k = util_tf2.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1]) # [k, c]
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets # [k, c]
        top_antecedents_mask = raw_top_antecedents >= 0 # [k, c]
        top_antecedents = tf.maximum(raw_top_antecedents, 0) # [k, c]

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores, top_antecedents) # [k, c]
        top_fast_antecedent_scores += tf.math.log(tf.cast(top_antecedents_mask, dtype=tf.float32)) # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_entity_scores(self, span_emb):
        """
        Args:
            span_emb: [k, emb]
        Returns:
            entity_scores: [k, entity_classes]
        """
        # every class have a FFNN
        classes_score = []
        for i in range(self.config['entity_classes']):
            with tf.compat.v1.variable_scope("entity_scores_{}".format(i)):
                class_score = util_tf2.ffnn(span_emb, self.config["entity_ffnn_depth"], self.config["entity_ffnn_size"], \
                    1, self.dropout) # [num_candidates, 1]
                classes_score.append(class_score)
        
        entity_scores = tf.concat(classes_score, 1)
        return tf.nn.softmax(entity_scores, 1)
    
    def get_entity_label_mask(self, entity_labels):
        """
        Args:
            entity_labels: [k]
        Returns:
            entity_labels_mask: [k, num_classes]
        """
        entity_index = tf.expand_dims(tf.range(self.config['entity_classes']), 0) # [1, num_classes]
        entity_labels = tf.expand_dims(entity_labels, 1) # [k, 1]
        entity_labels_mask = tf.equal(entity_index, entity_labels)
        return tf.cast(entity_labels_mask, dtype=tf.float32)

    def get_relation_scores(self, span_emb):
        """
        Args:
            span_emb: [k, emb]
        Returns:
            relation_scores: [k, k, relation_classes]
        """
        #k = util_tf2.shape(span_emb, 0)
        #offset = tf.expand_dim(tf.range(k), 1) + tf.expand_dim(tf.range(k), 0)

        pass

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.compat.v1.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util_tf2.projection(top_span_emb, util_tf2.shape(top_span_emb, -1)), 1 - (self.dropout)) # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, 1 - (self.dropout)) # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True) # [k, k]

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, \
        top_antecedent_offsets, top_span_speaker_ids, genre_emb):
        """
        Args:
            top_span_emb: [k, emb],
            top_antecedents: [k, c],
            top_antecedent_emb: [k, c, emb],
            top_antecedent_offsets: [k, c],
            top_span_speaker_ids: [k],
            genre_emb: genre embedding
        """
        k = util_tf2.shape(top_span_emb, 0)
        c = util_tf2.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents) # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids) # [k, c]
            speaker_pair_emb = tf.gather(tf.compat.v1.get_variable("same_speaker_emb", [2, self.config["feature_size"]]), \
                tf.cast(same_speaker, dtype=tf.int32)) # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1]) # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
            antecedent_distance_emb = tf.gather(tf.compat.v1.get_variable("antecedent_distance_emb", \
                [10, self.config["feature_size"]]), antecedent_distance_buckets) # [k, c]? [k, c, emb]?
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2) # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, 1 - (self.dropout)) # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1) # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1]) # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

        with tf.compat.v1.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util_tf2.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout) # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2) # [k, c]
        return slow_antecedent_scores # [k, c]

    def cnn(self, inputs, filter_sizes, num_filters):
        """ concatenate 3 conv1d layer output
        in_channel, out_channel
        """
        num_words = util_tf2.shape(inputs, 0)
        num_chars = util_tf2.shape(inputs, 1)
        input_size = util_tf2.shape(inputs, 2)

        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.variable_scope("conv_{}".format(i)):
                w = tf.compat.v1.get_variable("w", [filter_size, input_size, num_filters])
                b = tf.compat.v1.get_variable("b", [num_filters])
            conv = tf.nn.conv1d(input=inputs, filters=w, stride=1, padding="VALID") # [num_words, num_chars - filter_size, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, b)) # [num_words, num_chars - filter_size, num_filters]
            pooled = tf.reduce_max(input_tensor=h, axis=1) # [num_words, num_filters]
            outputs.append(pooled)
        
        return tf.concat(outputs, 1) # [num_words, num_filters * len(filter_sizes)]
    
    def batch_gather(self, emb, indices):
        batch_size = util_tf2.shape(emb, 0)
        seqlen = util_tf2.shape(emb, 1)
        if len(emb.get_shape()) > 2:
            emb_size = util_tf2.shape(emb, 2)
        else:
            emb_size = 1
        
        flattened_emb = tf.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
        offset = tf.expand_dims(tf.range(batch_size) * seqlen, 1)  # [batch_size, 1]
        gathered = tf.gather(flattened_emb, indices + offset) # [batch_size, num_indices, emb]
        if len(emb.get_shape()) == 2:
            gathered = tf.squeeze(gathered, 2) # [batch_size, num_indices]
        return gathered
    
    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        Args:
            distances: [k, c]
        """
        logspace_idx = tf.cast(tf.floor(tf.math.log(tf.cast(distances, dtype=tf.float32))/math.log(2)), dtype=tf.int32) + 3
        use_identity = tf.cast(distances <= 4, dtype=tf.int32)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)
    
    def softmax_loss(self, antecedent_scores, antecedent_labels):
        """
        Args:
            antecedent_scores: [k, c+1],
            antecedent_labels: [k, c+1]
        Returns:
            loss: [k]
        """
        gold_scores = antecedent_scores + tf.math.log(tf.cast(antecedent_labels, dtype=tf.float32)) # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(input_tensor=gold_scores, axis=[1]) # [k]
        log_norm = tf.reduce_logsumexp(input_tensor=antecedent_scores, axis=[1]) # [k]
        return log_norm - marginalized_gold_scores # [k]
    
    def entity_loss(self, entity_scores, entity_labels):
        """
        Args:
            entity_scores: [k, num_classes],
            entity_labels: [k, num_classes], binary matrix
        Returns:
            loss: [k]
        """
        self.score_tensor = entity_scores * entity_labels # [k, num_classes]
        return -tf.math.log(tf.reduce_sum(self.score_tensor, axis=1))

    ### evaluate ###
    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example
            
            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)
            print("Loaded {} eval examples.".format(len(self.eval_data)))

    

    def evaluate(self, session, official_stdout=False):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
            feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
            candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = \
                session.run(self.predictions, feed_dict=feed_dict)
            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, example["clusters"], coref_evaluator)
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        summary_dict = {}
        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        summary_dict["Average F1 (conll)"] = average_f1
        print("Average F1 (conll): {:.2f}%".format(average_f1))

        p,r,f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}%".format(f * 100))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        return util_tf2.make_summary(summary_dict), average_f1
    
    def evaluate_entity(self, session):
        self.load_eval_data()

        entity_evaluator = EntityEvaluator()

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
            entity_scores, entity_labels_mask = session.run(self.predictions, feed_dict=feed_dict)
            entity_evaluator.merge_input(entity_scores, entity_labels_mask)

            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num, len(self.eval_data)))
        
        return entity_evaluator.calc_f1(), entity_evaluator.calc_accuracy()

