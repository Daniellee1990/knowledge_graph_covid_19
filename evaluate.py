#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import entity_model as em
import util

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = em.EntityModel(config)
  with tf.Session() as session:
    model.restore(session)
    model.evaluate(session, official_stdout=True)
