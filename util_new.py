import keras
import numpy as np
import tensorflow_hub as hub
import os

from keras.models import Model
from keras import Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models,layers
from nltk import sent_tokenize, word_tokenize

def load_text(text_path):
    with open(text_path, encoding='UTF-8') as f:
        text = f.read().lower()
    
    sentences = [word_tokenize(s) for s in sent_tokenize(text)]
    num_sentences = len(sentences) # document中句子的总数
    max_sentence_length = max(len(s) for s in sentences) # 最长的句子长度（word-level）
    max_word_length = max(max(len(w) for w in s) for s in sentences) # 最长的单词长度（char-level）
    num_words = len(set(word_tokenize(text))) # document中出现的token的数量
    return sentences, num_sentences, max_sentence_length, max_word_length, num_words

def load_embedding_dict(glove_dir):
    """ 解析 GLOVE 词嵌入文件，返回 GLOVE 词嵌入字典 {单词 : 单词对应的embedding_vector}
    """
    embedding_dict = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='UTF-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs
    f.close()
    print("Done loading word embeddings.")
    print('Found %s word vectors.' % len(embedding_dict))
    return embedding_dict

def context_word(sentences):
    tokenizer = Tokenizer(num_words=num_words, filters="") # 初始化一个Tokenizer，考虑document中所有出现的token
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences) #　将字符串列表转换为整数索引列表
    max_sentence_length = max(len(s) for s in sequences)
    word_index = tokenizer.word_index # 保存每个 token 的整数索引
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) # 保存逆索引
    print('Found %s unique tokens.' % len(word_index)) # 检查token数量
    # num_words是使用nltk做tokenization得到的结果，len(word_index)是使用keras的tokenizer得到的结果，
    # num_words会覆盖len(word_index)，即 num_words >= len(word_index)
    context_word = pad_sequences(sequences, maxlen=max_sentence_length, padding='post')# 将整数索引列表转换为张量
    return context_word

def get_glove_embedding_matrix(max_words, embedding_dim):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in embedding_dict.items():
        embedding_vector = embedding_dict.get(word)
        if i < max_words:
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def char_dict():
    """ 制作字符表
    """ 
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for index, char in enumerate(alphabet):
        char_dict[char] = index + 1
    return char_dict

def context_char(sentences,num_sentences, max_sentence_length, max_word_length):
    context_char = np.zeros([num_sentences, max_sentence_length, max_word_length])
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            context_char[i, j, : ] = pad_sequences(tk.texts_to_sequences([word]), maxlen=max_word_length, padding='post').flatten()
    return context_char

class ElmoEmbeddingLayer():
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable = True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        #self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        #super(ElmoEmbeddingLayer, self).build(input_shape)
        
    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable, name="{}_module".format(self.name))

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                    as_dict=True,
                    signature='default',
                    )['default']
        return result
            
    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')
            
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

def glove_cnn_elmo_model(max_words, embedding_dim, max_sentence_length, num_sentences):
    # glove 部分
    word_input = Input(shape=(max_sentence_length, ), name='context_word')
    embedding_content_word = layers.Embedding(max_words, embedding_dim)(word_input)
    embedding_content_word.set_weights([embedding_matrix]) # 将预训练的词嵌入加载到 Embedding 层中
    embedding_content_word.trainable = False # 冻结 GLOVE_Embedding 层
    # char_cnn 部分
    char_input = Input(shape=(num_sentences*max_sentence_length, ), name='context_char')
    embedding_content_char = layers.Conv1D(50, 3)(char_input)
    embedding_content_char = layers.Conv1D(50, 4)(embedding_content_char)
    embedding_content_char = layers.Conv1D(50, 5)(embedding_content_char)
    embedding_content_char.reshape(num_sentences, max_sentence_length, -1)
    # glove cnn concatenation
    glove_cnn_concatenated = layers.concatenate([embedding_content_word, embedding_content_char], axis=-1)
    embedding = ElmoEmbeddingLayer()(glove_cnn_concatenated)
    return embedding

if __name__ == '__main__':
    file_path = 'test1.txt'
    sentences, num_sentences, max_sentence_length, max_word_length, num_words = \
        load_text(file_path)
    print(num_sentences)
    print(max_sentence_length)
    print(max_word_length)
