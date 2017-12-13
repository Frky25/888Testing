from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time, os

import numpy as np
import matplotlib.pyplot as plt

params = {
    'max_steps':30000,
    'batch_size': 50,
    'data_dir': 'input_data',
    'log_dir': 'logs'
}

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
#options are original, naive, high, low, and combo
dset = 'combo'
nch = 1
if dset == 'original':
    data_set = input_data.read_data_sets('input_data')
elif dset == 'naive':
    data_set_in = input_data.read_data_sets('input_data')
    images0 = data_set_in.train.images
    images1 = data_set_in.validation.images
    images2 = data_set_in.test.images
    for i in range(len(images0)):
        for j in range(len(images0[i])):
            if images0[i][j]>0.5:
                images0[i][j] = 1.
            else:
                images0[i][j] = 0.
    for i in range(len(images1)):
        for j in range(len(images1[i])):
            if images1[i][j]>0.5:
                images1[i][j] = 1.
            else:
                images1[i][j] = 0.
    for i in range(len(images2)):
        for j in range(len(images2[i])):
            if images2[i][j]>0.5:
                images2[i][j] = 1.
            else:
                images2[i][j] = 0.    
    d0 = DataSet(images0, data_set_in.train.labels, reshape = False, dtype=dtypes.uint8)
    d1 = DataSet(images1, data_set_in.validation.labels, reshape = False, dtype=dtypes.uint8)
    d2 = DataSet(images2, data_set_in.test.labels, reshape = False, dtype=dtypes.uint8)
    data_set = base.Datasets(train=d0, validation=d1, test=d2)
elif dset == 'low':
    dat_train = np.load('transformed_training_low.npy')
    labels_train = np.load('training_labels.npy')
    dat_test = np.load('transformed_testing_low.npy')
    labels_test = np.load('testing_labels.npy')
    images0 = dat_train.reshape((-1,28*28))
    images1 = dat_test.reshape((-1,28*28))
    images2 = dat_test.reshape((-1,28*28))
    d0 = DataSet(images0, labels_train, reshape = False, dtype=dtypes.uint8)
    d1 = DataSet(images1, labels_test, reshape = False, dtype=dtypes.uint8)
    d2 = DataSet(images2, labels_test, reshape = False, dtype=dtypes.uint8)
    data_set = base.Datasets(train=d0, validation=d1, test=d2)
elif dset == 'high':
    dat_train = np.load('transformed_training_high.npy')
    labels_train = np.load('training_labels.npy')
    dat_test = np.load('transformed_testing_high.npy')
    labels_test = np.load('testing_labels.npy')
    images0 = dat_train.reshape((-1,28*28))
    images1 = dat_test.reshape((-1,28*28))
    images2 = dat_test.reshape((-1,28*28))
    d0 = DataSet(images0, labels_train, reshape = False, dtype=dtypes.uint8)
    d1 = DataSet(images1, labels_test, reshape = False, dtype=dtypes.uint8)
    d2 = DataSet(images2, labels_test, reshape = False, dtype=dtypes.uint8)
    data_set = base.Datasets(train=d0, validation=d1, test=d2)
elif dset == 'combo':
    nch = 2
    dat_train1 = np.load('transformed_training_low.npy')
    dat_train2 = np.load('transformed_training_high.npy')
    labels_train = np.load('training_labels.npy')
    dat_test1 = np.load('transformed_testing_low.npy')
    dat_test2 = np.load('transformed_testing_high.npy')
    labels_test = np.load('testing_labels.npy')
    dat_train = np.concatenate((dat_train1,dat_train2),3)
    dat_test = np.concatenate((dat_test1,dat_test2),3)
    images0 = dat_train.reshape((-1,28*28*2))
    images1 = dat_test.reshape((-1,28*28*2))
    images2 = dat_test.reshape((-1,28*28*2))
    d0 = DataSet(images0, labels_train, reshape = False, dtype=dtypes.uint8)
    d1 = DataSet(images1, labels_test, reshape = False, dtype=dtypes.uint8)
    d2 = DataSet(images2, labels_test, reshape = False, dtype=dtypes.uint8)
    data_set = base.Datasets(train=d0, validation=d1, test=d2)
else:
    print ("Invalid dataset")
    quit()
    
def f1(x, filter_height, filter_width, in_chan, out_chan, name):
    with tf.name_scope(name) as scope:
        W = tf.get_variable(name+'_w', shape = [filter_height, filter_width, in_chan, out_chan])
        b = tf.get_variable(name+'_b', shape = [out_chan])  

        c = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        y = tf.nn.bias_add(c, b, name=scope)
    return y

def f2(x, name):
    with tf.name_scope(name) as scope:
        y = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=scope)
    return y

def f3(x, name):
    with tf.name_scope(name) as scope:
        y = tf.nn.relu(x, name=scope)
    return y

def f4(x, num_in, num_out, name):
    with tf.name_scope(name) as scope:
        # Create tf variables for the weights and biases
        W = tf.get_variable(name+'_w', shape=[num_in, num_out], trainable=True)
        b = tf.get_variable(name+'_b', [num_out], trainable=True)
        y = tf.nn.bias_add(tf.matmul(x, W), b, name=scope)
    return y

def flatten(x, num_out):
    return tf.reshape(x, [-1, num_out])

def training_graph():
    dcn_graph = tf.Graph()
    with dcn_graph.as_default():
        inputs = tf.placeholder(tf.float32, [params['batch_size'], 784*nch], name="inputs")
        labels = tf.placeholder(tf.int32, params['batch_size'], name="labels")

        # Setup the inference nodes
        with tf.name_scope('inference') as scope:
            input_image = tf.reshape(inputs, [-1,28,28,nch], name="input_image")
            l1 = f1(input_image, 5, 5, nch, 4, "l1")
            l2 = f3(l1, "l2")
            l3 = f2(l2, "l3")
            l4 = f1(l3, 5, 5, 4, 8, "l4")
            l5 = f3(l4, "l5")
            l6 = f2(l5, "l6")
            l7 = f4(flatten(l6, 392), 392, 256, "l7")
            l8 = f3(l7, "l8")
            l9 = f4(l8, 256, 10, "l9")
            logits = tf.identity(l9, name=scope)

        # Setup the loss and optimizer nodes
        with tf.name_scope('loss') as scope:
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name=scope)

        with tf.name_scope('train_step') as scope:
            optimizer = tf.train.AdamOptimizer(1e-4)
            train_step = optimizer.minimize(loss, name=scope)

        # Setup evaluation nodes
        with tf.name_scope('eval_step') as scope:
            correct = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.int32)
            # Return the number of true entries.
            eval_correct = tf.reduce_sum(correct, name=scope)

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('eval_correct', eval_correct)
        return dcn_graph
    
dcn_graph = training_graph()

def eval_net(sess, data_set, params):
    steps_per_epoch = data_set.num_examples // params['batch_size']
    num_examples = steps_per_epoch * params['batch_size']

    model_graph = sess.graph
    inputs = model_graph.get_operation_by_name('inputs').outputs[0]
    labels = model_graph.get_operation_by_name('labels').outputs[0]
    eval_step = model_graph.get_operation_by_name('eval_step').outputs[0]

    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    for step in range(steps_per_epoch):
        inputs_ds, labels_ds = data_set.next_batch(params['batch_size'])
        feed_dict = { inputs: inputs_ds, labels: labels_ds }
        true_count += sess.run(eval_step, feed_dict=feed_dict)
    accuracy = float(true_count) / num_examples
    return accuracy
    
def train_net(sess, data_set, params):
    # Update this function to also return the validation error at each checkpoint
    # so that you can plot it later.
    
    # Define the network and setup training parameters
    model_graph = sess.graph
    inputs = model_graph.get_operation_by_name('inputs').outputs[0]
    labels = model_graph.get_operation_by_name('labels').outputs[0]
    train_step = model_graph.get_operation_by_name('train_step')
    loss = model_graph.get_operation_by_name('loss').outputs[0]

    log_interval, checkpoint_interval = 100, 1000
    step_index = np.zeros(int(params['max_steps']/checkpoint_interval))
    train_error = np.zeros(int(params['max_steps']/checkpoint_interval))
    validation_error = np.zeros(int(params['max_steps']/checkpoint_interval))

    train_epoch = 0
    with sess.graph.as_default():
        # Logging and initialization before starting a session
        saver = tf.train.Saver()
        summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(params['log_dir'], sess.graph)
        sess.run(tf.global_variables_initializer())

        for step in range(params['max_steps']):
            start_time = time.time()

            inputs_ds, labels_ds = data_set.train.next_batch(params['batch_size'])
            feed_dict = { inputs: inputs_ds, labels: labels_ds }
            _, loss_value = sess.run([train_step, loss], feed_dict=feed_dict)

            duration = time.time() - start_time

            # Write the summaries and print an overview fairly often.
            if step % log_interval == 0:
                # Print status to stdout.
                # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (step + 1) % checkpoint_interval == 0 or (step + 1) == params['max_steps']:
                checkpoint_file = os.path.join(params['log_dir'], 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                step_index[train_epoch] = step+1
                train_error[train_epoch] = 1 - eval_net(sess, data_set.train, params)
                validation_error[train_epoch] = 1 - eval_net(sess, data_set.validation, params)
                
                print("%d %g %g" % (step_index[train_epoch], train_error[train_epoch], validation_error[train_epoch]))
                train_epoch += 1
    return (step_index, train_error, validation_error)
    
sess = tf.Session(graph=dcn_graph)
step_index, train_error, validation_error = train_net(sess, data_set, params)