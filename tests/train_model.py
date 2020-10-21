#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import math

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

    log_dir = config["log_dir"]
    writer = tf.compat.v1.summary.FileWriter(log_dir, flush_secs=30, filename_suffix='train')

    max_f1 = 0
    max_acc = 0

    # with tf.compat.v1.Session() as session:
    with tf.compat.v1.Session().as_default() as session:
        model.start(session)
        model.train()
        saver = tf.compat.v1.train.Saver()
        
        accumulated_loss = 0.0

        #ckpt = tf.train.get_checkpoint_state(log_dir)
        #if ckpt and ckpt.model_checkpoint_path:
        #    print("Restoring from: {}".format(ckpt.model_checkpoint_path))
        #    saver.restore(session, ckpt.model_checkpoint_path)
        
        session.run(tf.compat.v1.global_variables_initializer())

        initial_time = time.time()
        LIMIT = 7000
        while True:
            #print(123)
            #time.sleep(5)
            tf_loss, tf_global_step, relation_labels_mask, relation_scores, _ = \
                session.run([model.loss, model.global_step, model.relation_labels_mask, model.relation_scores, model.train_op])
            accumulated_loss += tf_loss

            #print('test labels\n', tf_coref_labels)
            print('training literature: {}'.format(tf_global_step))
            #print('test relations')
            #print(relation_scores.shape, relation_labels_mask.shape)
            #print(relation_labels)

            print('test loss', tf_loss)
            for i in range(relation_scores.shape[0]):
                for j in range(relation_scores.shape[1]):
                    if math.isnan(sum(relation_scores[i][j])):
                        print('error!!!', relation_scores[i][j])

            if tf_global_step > 0 and tf_global_step % report_frequency == 0:
                total_time = time.time() - initial_time
                steps_per_second = tf_global_step / total_time

                average_loss = accumulated_loss / report_frequency
                print("[{}] loss={:.2f}, steps/s={:.2f}".format(tf_global_step, average_loss, steps_per_second))
                writer.add_summary(util_tf2.make_summary({"loss": average_loss, "global_step": tf_global_step}), tf_global_step)
                accumulated_loss = 0.0

            # evaluate
            if tf_global_step > 0 and tf_global_step % eval_frequency == 0:
                # eval_summary, eval_f1 = model.evaluate(session)
                #eval_f1, eval_acc = model.evaluate_entity(session)
                eval_f1, eval_acc = model.evaluate_relation(session)

                if eval_f1 > max_f1:
                    max_f1 = eval_f1
                    #saver.save(session, os.path.join(log_dir, "model"), global_step=tf_global_step)
                    #util_tf2.copy_checkpoint(os.path.join(log_dir, "model-{}".format(tf_global_step)), os.path.join(log_dir, "model.max.ckpt"))
                if eval_acc > max_acc:
                    max_acc = eval_acc

                #writer.add_summary(eval_summary, tf_global_step)
                writer.add_summary(util_tf2.make_summary({"max_eval_f1": max_f1, "max_acc": max_acc}), tf_global_step)
                print("[{}] evaL_f1={:.2f}, max_f1={:.2f}, max_acc={:.2f}".format(tf_global_step, eval_f1, max_f1, max_acc))

            if tf_global_step == LIMIT:
                print('Training Done')
                break
