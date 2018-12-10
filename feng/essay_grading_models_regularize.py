import tensorflow as tf
import numpy as np
import time, os
import matplotlib.pyplot as plt
import csv, math

import pandas as pd
from collections import Counter

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# note:
# *1. loss function: look up the data mining book
# 2. fully_connected regression point of view
# 3. loss function: regularization
# _4. unpadding
# 5. bidirectional
# *6. deep LSTM
# 7. validation to find the best parameter
# *8. LSTMCell LSTMBlockCell LSTMBLockFusedCell LSTMBlockWrapper?
# 9. find the frequency of different label, use 1/freq to reweigh the loss in order to augment less frequent label's learning

import sklearn.metrics

from helper import Helper
import utils


# Read in data
# X_, _, y_ = Helper(set_num=0, file_name='../data/small.tsv').get_embed()
X_, _, y_ = Helper(set_num=0, file_name='../data/training_set_rel3.tsv').get_embed()

_, inv_freq = utils.hist_freq(y_, 12)

data_size = y_.shape[0]
train_size = math.floor(0.9 * data_size)
test_size = data_size - train_size

X_train = X_[:train_size]
y_train = y_[:train_size]
X_test = X_[train_size:]
y_test = y_[train_size:]


# X_train is embedded essays (num_essays, num_word, embedding_dim = 100)
# y_train is the corresponding labels ([num_essays])



# 3 constant
N_CLASSES = 12
# N_HIDDEN = 128

n_hidden_in = 128
n_hidden_out = 64

# lstm_sizes = [n_hidden_in, n_hidden_out]
# lstm_sizes = [128, 64, 32]
deep_nets = [[128], [128, 64], [128, 64, 32], [128, 64, 128, 64, 32]]
lstm_sizes = deep_nets[2]

multilayer = ""
if len(lstm_sizes) > 1:
    multilayer = "multi"

cells = ['lstm', 'lstm_block', 'lstm_block_fused', 'gru']
controllers = ['one_direction', 'bidirection']
cell_type = cells[3]
controller_type = controllers[1]

# Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DISP_STEP = 1
DROPOUT = 0.5
N_EPOCHS = 50
l2_reg_lambda = 1.0

embedding_size = 100
# maxlength = 800


bidi = tf.constant(1, dtype=tf.int32)
if controller_type == "bidirection":
    bidi = tf.constant(2, dtype=tf.int32)

np.random.seed(0)



print(cell_type)
print(controller_type)
print(lstm_sizes)



# In[4]
# Utils:
def one_hot(Y, n_classes):
    batch_size = Y.shape[0]
    output = np.zeros((batch_size, n_classes), dtype=int)
    # for idx in range(batch_size):
    #     output[idx, Y[idx]-1] = 1

    output[np.arange(batch_size), Y.astype(int) - 1] = 1
    # return output
    return output


def one_hot_reverse(Y):
    output = np.argwhere(Y == 1)[:, 1] + 1

    return output


def batch_zero_pad(x_batch):
    """
    Zero Pad input messages
    :param x_batch: Input list of encoded messages (num_batch, num_word, embedding_dim)
    # :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The padded essays. (num_batch, maxlength, embedding_dim)
    """

    maxlength = 0
    for essay in x_batch:
        if essay.shape[0] > maxlength:
            maxlength = essay.shape[0]

    embedding_size = x_batch[0].shape[1]
    x_batch_padded = np.zeros((len(x_batch), maxlength, embedding_size))
    for idx, essay in enumerate(x_batch):
        non_empty_length = essay.shape[0]
        x_batch_padded[idx, :non_empty_length] = essay

    return x_batch_padded, maxlength
# test: X_batch = batch_zero_pad(X_train[:100])


Y_train = one_hot(y_train, N_CLASSES)


#In[5]

with tf.name_scope('data'):
    # num_word = tf.Variable(tf.constant(800, tf.int32), name='num_word')
    seqlen = tf.placeholder(tf.int32, [BATCH_SIZE], name='sequence_len')
    X = tf.placeholder(tf.float32, [BATCH_SIZE, None, 100], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, N_CLASSES], name="Y_placeholder")

    # state = tf.placeholder(tf.float32, shape=[None_preds, 2*N_HIDDEN])
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    l2_loss = tf.constant(0.0)
    loss_weight = tf.placeholder(tf.float32, [N_CLASSES])


#In[6]
with tf.name_scope('process_data'):
    w1 = tf.Variable(tf.truncated_normal(shape=[100, lstm_sizes[0]], stddev=1.0), name='w1')
    b1 = tf.Variable(tf.truncated_normal([lstm_sizes[0]], mean=0.0, stddev=1.0), name='b1')

    # _X = tf.transpose(X, perm=[1, 0, 2]) # >> _X.shape = num_steps, BATCH_SIZE, 100
    # _X = tf.reshape(_X, shape=[-1, 100]) # >> _X.shape = num_steps * BATCH_SIZE, 100
    # _X = tf.nn.xw_plus_b(_X, w1, b1)
    # _X = tf.split(_X, _X.shape[0], 0) # _X.shape = [ (BATCH_SIZE, 100) for _ in range(num_steps)]

    _X = tf.add(tf.tensordot(X, w1, [[2], [0]]), b1)  # >> X.shape = BATCH_SIZE, num_steps, 100

    # _X = tf.nn.dropout(_X, keep_prob=dropout_keep_prob)

    l2_loss += tf.nn.l2_loss(w1)
    l2_loss += tf.nn.l2_loss(b1)





def get_cell(hidden_size, cell_type):
    if cell_type == "lstm":
        return tf.nn.rnn_cell.LSTMCell(hidden_size, forget_bias=1.0)
    # This is added to biases the forget gate
    # in order to reduce the scale of forgetting in the beginning of the training.

    elif cell_type == "lstm_block":
        return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=1.0)

    # elif cell_type == "lstm_block_fused":
    #     return tf.contrib.rnn.LSTMBlockFusedCell(hidden_size)
    # not supported

    elif cell_type == "gru":
        return tf.nn.rnn_cell.GRUCell(hidden_size)

    else:
        print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
        return None


def get_controller(cell, _X, seqlen, controller_type="one_direction"):
    batch_size = _X.shape[0]
    if controller_type == "one_direction":
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
        last_output = tf.gather_nd(outputs, tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))
        # which is the Tensorflow equivalent of
        #  numpy's rnn_outputs[range(30), seqlen-1, :]

        return last_output

    elif controller_type == "bidirection":
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
        last_outputs = [tf.gather_nd(outputs[i], tf.stack([tf.range(batch_size), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1)) for i in range(2)]
        # which is the Tensorflow equivalent of
        #  numpy's rnn_outputs[range(30), seqlen-1, :]

        return tf.concat(last_outputs, axis=1)

    else:
        print("ERROR: '" + controller_type + "' is a wrong controller type. Use default.")
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
        last_output = tf.gather_nd(outputs, tf.stack([tf.range(BATCH_SIZE), tf.maximum(seqlen - 1, tf.zeros([batch_size], dtype=tf.int32))], axis=1))
        # which is the Tensorflow equivalent of
        #  numpy's rnn_outputs[range(30), seqlen-1, :]

        return last_output




def model(_X, seqlen , lstm_sizes, dropout_keep_prob, cell_type="lstm", controller_type="one_direction"):

    lstms = [get_cell(lstm_size, cell_type=cell_type) for lstm_size in lstm_sizes ]
    drops = [tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    last_output = get_controller(cell, _X, seqlen, controller_type=controller_type)



    # # lstm_cell = tf.contrib.rnn.LSTMCell(N_HIDDEN, forget_bias=1.0, state_is_tuple=True)  # attention: state_is_tuple
    # lstm_cell = get_cell(lstm_sizes, cell_type=cell_type)
    #
    # cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
    # # Drop out is applied on a output neuron to randomly shut it down, in order to reduce the compacity of the model.
    # # Simultanously, since each neuron outputs zero with prob= (1-keep_prob), when testing it, the weight w shoud be times by the prob)
    #
    # # DropoutWrapper is to mask all the ends of a cell, it uses dropout internally.
    # # dropout is to mask only one end
    #
    # last_output = get_controller(cell, _X, seqlen, controller_type=controller_type)
    #
    # # last_output = tf.nn.dropout(last_output, keep_prob=dropout_keep_prob)
    # # This should be equivalent to above DropoutWrapper

    return last_output




with tf.name_scope('lstm') as scope:
    # lstm_cell = tf.contrib.rnn.LSTMCell(N_HIDDEN, forget_bias=1.0, state_is_tuple=True) # attention: state_is_tuple
    # cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=dropout_keep_prob)
    # outputs, _ = tf.nn.dynamic_rnn(cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
    # # outputs_bidi, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs=_X, dtype=tf.float32, sequence_length=seqlen, time_major=False)
    # last_output = tf.gather_nd(outputs, tf.stack([tf.range(BATCH_SIZE), seqlen - 1], axis=1))
    #
    # # which is the Tensorflow equivalent of numpy's rnn_outputs[range(30), seqlen-1, :]


    # last_output = model(_X, seqlen, [N_HIDDEN], dropout_keep_prob, cell_type, controller_type)
    last_output = model(_X, seqlen, lstm_sizes, dropout_keep_prob, cell_type, controller_type)



with tf.variable_scope('Output_layer') as scope:
    w2 = tf.Variable(tf.truncated_normal(shape=[lstm_sizes[-1] * bidi, N_CLASSES]), name='w2')
    # b2 = tf.Variable(tf.constant(0.1, shape=[N_CLASSES]), name='b2')
    b2 = tf.Variable(tf.truncated_normal([N_CLASSES], mean=0.0, stddev=1.0), name='b2')

    logits = tf.nn.xw_plus_b(last_output, w2, b2, name='logits')

    l2_loss += tf.nn.l2_loss(w2)
    l2_loss += tf.nn.l2_loss(b2)


with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name='loss') + l2_reg_lambda * l2_loss

    # prob_loss = tf.nn.softmax(logits) # 1* 1:12
    # score_candi_loss = tf.ones((BATCH_SIZE, 1)) * tf.cast(tf.range(N_CLASSES) + 1, tf.float32) # batch_size * 1:12
    # a_term = (score_candi_loss - Y) * Y   # Y batch_size * 1:12
    # a_term = tf.square(a_term) * loss_weight # loss_weight 1:12
    # loss = tf.reduce_sum(tf.multiply(prob_loss, a_term), name='lossFunction')


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


def batch_padding(X_test, y_test, test_size, batch_size):
    y = np.zeros(batch_size, dtype=int)
    y[:test_size] = y_test

    to_append = np.zeros_like(X_test[0])
    for idx in range(test_size, batch_size):
        X_test.append(to_append)

    return X_test, y

#In[28]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./graphs/aes', sess.graph)
    writer.close()
    ##### You have to create folders to store checkpoints
    # ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    # if ckpt and ckpt.model_checkpoint_path:
    #    saver.restore(sess, ckpt.model_checkpoint_path)

    # initial_step = global_step.eval()

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
            X_batch, num_word = batch_zero_pad(X_batch)
            # assert Y_batch.shape[0] == BATCH_SIZE:
            Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

            # print("Prior weight1 = ", w1.eval())
            # print("Prior weight2 = :", w2.eval())

            _, loss_batch, _accuracy, _pred_class, _last_output = sess.run([optimizer, loss, accuracy, pred_class, last_output ],
                        feed_dict={X: X_batch, Y: Y_batch,
                                   dropout_keep_prob: DROPOUT,
                                   seqlen: _seqlen,
                                   query_size: BATCH_SIZE,
                                   loss_weight: inv_freq})

            # print("Post weight1 = ", w1.eval())
            # print("Post weight2 = :", w2.eval())


            total_loss += loss_batch

            y_score = one_hot_reverse(Y_batch)
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

    # # test the model
    # n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    # total_correct_preds = 0
    # for i in range(n_batches):
    #     batch_zero_pad(X_batch), Y_batch = mnist.test.next_batch(BATCH_SIZE)
    #     _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
    #                                            feed_dict={X: batch_zero_pad(X_batch), Y: Y_batch, dropout: DROPOUT})
    #     total_correct_preds += sess.run(accuracy)
    #
    # print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))

    # test:
    # X_test, _, y_test = Helper(set_num=0, file_name='../data/test.tsv').get_embed()


    # test_size = y_test.shape[0]

    # if test_size < BATCH_SIZE:
    #     _, y_test = batch_padding(X_test, y_test, test_size, BATCH_SIZE)

    Y_test = one_hot(y_test, N_CLASSES)

    test_machine_score = np.array([])
    test_human_score = np.array([])
    count = 0

    for X_batch, Y_batch, _seqlen, effective_size in utils.get_batches(X_test, Y_test, BATCH_SIZE, needall='True'):
        X_batch, num_word = utils.batch_zero_pad(X_batch)
        Y_batch = Y_batch.reshape((BATCH_SIZE, N_CLASSES))

        _accuracy, _pred_class = sess.run(
            [accuracy, pred_class],
            feed_dict={X: X_batch, Y: Y_batch, dropout_keep_prob: DROPOUT, seqlen: _seqlen, query_size: effective_size, loss_weight: inv_freq})

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

    output_file_name = './files_to_plot/lstm' + '_' + cell_type + '_' + controller_type + '_' + multilayer + '.csv'
    with open(output_file_name, 'w', newline='\n') as fw:
        writer = csv.writer(fw, delimiter=',')
        writer.writerow(['test accuracy', _accuracy, 'kappa_test', kappa_test, 'tot_time', time.time()-start_time])
        writer.writerow(['machine_score'])
        writer.writerow(machine_score)
        writer.writerow(['human_score'])
        writer.writerow(human_score)
        writer.writerow(lstm_sizes)
        writer.writerow([''])


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

output_file_name = './files_to_plot/lstm0' +'_' + cell_type + '_' + controller_type +'_' +multilayer +'.csv'
for ord in range(1000):
    if os.path.exists('./files_to_plot/lstm' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_'+ str(ord) + '.csv'):
        continue
    else:
        output_file_name = './files_to_plot/lstm' +'_' + cell_type + '_' + controller_type +'_'+multilayer +'_' + str(ord) + '.csv'
        break

data_write_csv(output_file_name, [loss_fn, accuracies, kappa_scors])
# data_write_csv(output_file_name, [accuracies])
# data_write_csv(output_file_name, [kappa_scors])

plt.figure()
plt.plot(loss_fn)
plt.title('loss_t' + "_"+ cell_type + "_"+ controller_type)
plt.xlabel('#batch')

plt.figure()
plt.plot(accuracies)
plt.title('training_accuracy'+ "_"+ cell_type + "_"+ controller_type)
plt.xlabel('batch#')

plt.figure()
plt.plot(kappa_scors)
plt.title('kappa, tot_time = {}'.format(time.time() - start_time) + "_"+ cell_type + "_"+ controller_type)
plt.xlabel('#epochs')

plt.figure()
plt.plot(times)
plt.title('times')
plt.xlabel('#batch')

plt.show()