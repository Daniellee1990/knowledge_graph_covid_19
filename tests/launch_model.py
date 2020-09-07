#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.python.keras.backend import set_session
import entity_model_tf2 as em
import util_tf2

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    config = util_tf2.initialize_from_env()
    
    report_frequency = config["report_frequency"]
    eval_frequency = config["eval_frequency"]

    model = em.EntityModel(config)
    
    #saver = tf.compat.v1.train.Saver()

    log_dir = config["log_dir"]
    writer = tf.compat.v1.summary.FileWriter(log_dir, flush_secs=20)

    max_f1 = 0

    # with tf.compat.v1.Session() as session:
    global session
    with tf.compat.v1.Session().as_default() as session:
        #session.run(tf.compat.v1.global_variables_initializer())
        # model.start_enqueue_thread(session)
        model.start(session)
        accumulated_loss = 0.0

        #ckpt = tf.train.get_checkpoint_state(log_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    print("Restoring from: {}".format(ckpt.model_checkpoint_path))
        #    saver.restore(session, ckpt.model_checkpoint_path)

        initial_time = time.time()
        while True:
            #print(123)
            #time.sleep(5)
            
            #input_tensors = session.run(model.input_tensors)
            model.train()
            #
            tf_loss, tf_global_step, _ = session.run([model.loss, model.global_step, model.train_op])
            #tf_loss, tf_global_step = session.run([model.loss, model.global_step])
            accumulated_loss += tf_loss
            print('test', accumulated_loss)

            if tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                writer.add_summary(util_tf2.make_summary({"loss": average_loss}), tf_global_step)
                accumulated_loss = 0.0
            #'''
            if tf_global_step % eval_frequency == 0:
                saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                eval_summary, eval_f1 = model.evaluate(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    util_tf2.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))

                #writer.add_summary(eval_summary, tf_global_step)
                #writer.add_summary(util_tf2.make_summary({"max_eval_f1": max_f1}), tf_global_step)
                print("[{}] evaL_f1={:.2f}, max_f1={:.2f}".format(tf_global_step, eval_f1, max_f1))
            #'''
