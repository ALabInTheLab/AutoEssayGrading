import tensorflow as tf
import numpy as np
import time, os
import matplotlib.pyplot as plt
import csv, math

import pandas as pd

import sklearn.metrics

from helper import Helper
import utils

N_CLASSES = 12
N_HIDDEN1 = 128
N_HIDDEN2 = 64

# Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DISP_STEP = 1
DROPOUT = 0.75
N_EPOCHS = 200

embedding_size = 100

np.random.seed(0)

# Read in data
# X_, _, y_ = Helper(set_num=0, file_name='../data/small.tsv').get_embed()
X_, y_ = Helper(set_num=0, file_name='../data/training_set_rel3.tsv').getAverage(embedding_size)

data_size = y_.shape[0]
train_size = math.floor(0.9 * data_size)
test_size = data_size - train_size

shuf_idx = np.random.permutation(data_size)
X_train = X_[shuf_idx][:train_size]
y_train = y_[shuf_idx][:train_size]
X_test = X_[shuf_idx][train_size:]
y_test = y_[shuf_idx][train_size:]


# X_train is embedded essays (num_essays, num_word, embedding_dim = 100)
# y_train is the corresponding labels ([num_essays])

Y_train = utils.one_hot(y_train, N_CLASSES)


with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [BATCH_SIZE, embedding_size], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name="Y_placeholder")


dropout = tf.placeholder(tf.float32, name='dropout')
dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.variable_scope('Layer1') as scope:
    w1 = tf.Variable(tf.truncated_normal(shape=[embedding_size, N_HIDDEN1]), name='w1')
    b1 = tf.Variable(tf.truncated_normal([N_HIDDEN1], mean=0.0, stddev=1.0), name='b1')

    output1 = tf.nn.xw_plus_b(X, w1, b1, name='logits')

    output1 = tf.nn.relu(output1, name='relu')
    #output1 = tf.nn.dropout(output1, dropout, name='dropout')

with tf.variable_scope('Layer2') as scope:
    w2 = tf.Variable(tf.truncated_normal(shape=[N_HIDDEN1, N_HIDDEN2]), name='w2')
    b2 = tf.Variable(tf.truncated_normal([N_HIDDEN2], mean=0.0, stddev=1.0), name='b2')

    output2 = tf.nn.xw_plus_b(output1, w2, b2, name='logits')

    output2 = tf.nn.relu(output2, name='relu')
    #output2 = tf.nn.dropout(output2, dropout, name='dropout')

with tf.variable_scope('Layer3') as scope:
    w3 = tf.Variable(tf.truncated_normal(shape=[N_HIDDEN2, N_CLASSES]), name='w3')
    b3 = tf.Variable(tf.truncated_normal([N_CLASSES], mean=0.0, stddev=1.0), name='b3')

    logits = tf.nn.xw_plus_b(output2, w3, b3, name='logits')

with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss')


with tf.name_scope('optimizer') as scope:
    #: define training op
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

#In[23]
with tf.name_scope('accuracy') as scope:
    query_size = tf.placeholder(dtype=tf.int32, shape=None)

    score_candi = tf.ones((query_size, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32)
    prob = tf.nn.softmax(logits[:query_size])
    pred_class = tf.reduce_sum(tf.multiply(prob, score_candi), 1)  ## attention: score_candi lazy loading?

    # correct_preds = tf.equal(tf.to_int64(tf.round(pred_class)), tf.argmax(Y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    # correct_preds = tf.math.exp(- tf.abs((tf.round(pred_class) - tf.cast(tf. argmax(Y, 1), tf.float32) )))
    correct_preds = tf.math.exp(- 0.5 * tf.square((tf.round(pred_class) - tf.cast(tf. argmax(Y[:query_size], 1), tf.float32) )) / 3)
    accuracy = tf. reduce_mean(tf.cast(correct_preds, tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs/aes', sess.graph)
    writer.close()

    start_time = time.time()
    tic_time = time.time()

    total_loss = 0.0
    index_batch = 0
    total_correct_preds = 0

    kappa_scors = []
    accuracies = []
    loss_fn = []
    times = []

    for i in range(N_EPOCHS):  # train the model n_epochs times
        machine_score = np.array([])
        human_score = np.array([])

        for X_batch, Y_batch, _seqlen, _ in utils.get_batches(X_train, Y_train, BATCH_SIZE):
            #X_batch, num_word = utils.batch_zero_pad(X_batch)
            # assert Y_batch.shape[0] == BATCH_SIZE:
            Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

            # print("Prior weight1 = ", w1.eval())
            # print("Prior weight2 = :", w2.eval())

            _, loss_batch, _accuracy, _pred_class, _logits = sess.run([optimizer, loss, accuracy, pred_class, logits ],
                                                        feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT, query_size: BATCH_SIZE})

            # print("Post weight1 = ", w1.eval())
            # print("Post weight2 = :", w2.eval())


            total_loss += loss_batch

            y_score = utils.one_hot_reverse(Y_batch)
            # print("Labels: ", y_score)

            machine_score = np.append(machine_score, _pred_class, axis=0)
            human_score = np.append(human_score, y_score, axis=0)

            # print('Predictions:', machine_score)
            print('Accuracy:', _accuracy)
            index_batch += 1

            # print("global_step == index_batch? ", global_step.eval() == index_batch)

            if (index_batch + 1) % DISP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index_batch + 1, total_loss / DISP_STEP))

                loss_fn.append(total_loss / DISP_STEP)
                accuracies.append(_accuracy)
                times.append(time.time() - tic_time)
                tic_time = time.time()

                total_loss = 0.0
                # saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index_batch)

        print("machine score:", np.round(machine_score).astype(int))
        print("human score:", human_score)

        score = sklearn.metrics.cohen_kappa_score(np.round(machine_score), np.round(human_score), weights='quadratic')
        print("Kappa score: ", score)
        print("epoch: {}".format(i))


        kappa_scors.append(score)


    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

# test:

    Y_test = utils.one_hot(y_test, N_CLASSES)

    test_machine_score = np.array([])
    test_human_score = np.array([])
    count = 0

    for X_batch, Y_batch, _seqlen, effective_size in utils.get_batches(X_test, Y_test, BATCH_SIZE, needall=False):
        #X_batch, num_word = utils.batch_zero_pad(X_batch)
        Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

        _accuracy, _pred_class = sess.run(
            [accuracy, pred_class],
            feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT, query_size: effective_size})

        y_score = utils.one_hot_reverse(Y_batch[:effective_size])

        test_machine_score = np.append(test_machine_score , _pred_class)
        test_human_score = np.append(test_human_score, y_score)
        assert _pred_class.shape[0] == effective_size
        assert len(y_score) == effective_size

    kappa_test = sklearn.metrics.cohen_kappa_score(np.round(test_machine_score), np.round(test_human_score), weights='quadratic')


    print('count=', count)
    count += 1

    print('Test accuracy: ', _accuracy)
    print('Test kappa: ', kappa_test)

    output_file_name = './2layerNN.csv'
    with open(output_file_name, 'w+', newline='\n') as fw:
        writer = csv.writer(fw, delimiter=',')
        writer.writerow(['test accuracy', _accuracy, 'kappa_test', kappa_test, 'tot_time', time.time()-start_time])
        writer.writerow(['test_machine_score'])
        writer.writerow(test_machine_score)
        writer.writerow(['test_human_score'])
        writer.writerow(test_human_score)
        writer.writerow(['last epoch train machine_score'])
        writer.writerow(machine_score)
        writer.writerow(['train human_score'])
        writer.writerow(human_score)
        writer.writerow([])



## Output result:

def data_write_csv(file_name, datas):
    with open(file_name, 'w+', newline='\n') as file_csv:
    # file_csv = open(file_name,'w+', newline='\n')
        writer = csv.writer(file_csv, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for data in datas:
            writer.writerow(data)
        # writer.write('\n')
    print("save complete")
    print('wrote to' + file_name)

try:
    os.mkdir('files_to_plot')
except:
    pass

output_file_name = './files_to_plot/2layerNN.csv'


data_write_csv(output_file_name, [loss_fn, accuracies, kappa_scors])
# data_write_csv(output_file_name, [accuracies])
# data_write_csv(output_file_name, [kappa_scors])

plt.figure()
plt.plot(loss_fn)
plt.title('loss_t')
plt.xlabel('#batch')

plt.figure()
plt.plot(accuracies)
plt.title('training_accuracy')
plt.xlabel('batch#')

plt.figure()
plt.plot(kappa_scors)
plt.title('kappa, tot_time = {}'.format(time.time() - start_time))
plt.xlabel('#epochs')

plt.figure()
plt.plot(times)
plt.title('times')
plt.xlabel('#batch')

plt.show()