#!/usr/bin/env python
import os
import time
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

indices = [[1, 1]]
values = [1.0]
shape = [3, 3]
sp_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)

result = tf.sparse.to_dense(sp_tensor)

with tf.compat.v1.Session() as sess:
    x = sess.run([result])
    print('test x: {}'.format(x))