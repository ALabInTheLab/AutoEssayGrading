import numpy as np
import pandas as pd
import tensorflow as tf
import utils as utl
from collections import Counter

# Preprocess Data
# read data from csv file
data = pd.read_csv("data/StockTwits_SPY_Sentiment_2017.gz",
                   encoding="utf-8",
                   compression="gzip",
                   index_col=0)

# get messages and sentiment labels
messages = data.message.values
labels = data.sentiment.values

# View sample of messages with sentiment

for i in range(10):
    print("Messages: {}...".format(messages[i]),
          "Sentiment: {}".format(labels[i]))


messages = np.array([utl.preprocess_ST_message(message) for message in messages])

full_lexicon = " ".join(messages).split()
vocab_to_int, int_to_vocab = utl.create_lookup_tables(full_lexicon)

messages_lens = Counter([len(x) for x in messages])
print("Zero-length messages: {}".format(messages_lens[0]))
print("Maximum message length: {}".format(max(messages_lens)))
print("Average message length: {}".format(np.mean([len(x) for x in messages])))

messages, labels = utl.drop_empty_messages(messages, labels)

messages = utl.encode_ST_messages(messages, vocab_to_int)
labels = utl.encode_ST_labels(labels)

messages = utl.zero_pad_messages(messages, seq_len=244)


train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(messages, labels, split_frac=0.80)

print("Data Set Size")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))



# Build LSTM:

def model_inputs():
    """
    Create the model inputs
    """
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')

    return inputs_, labels_, keep_prob_


def build_embedding_layer(inputs_, vocab_size, embed_size):
    """
    Create the embedding layer
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

    return embed



def build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size):
    """
    Create the LSTM layers
    """
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)

    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

    return initial_state, lstm_outputs, cell, final_state


def build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate):
    """
    Create the Loss function and Optimizer
    """
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)
    loss = tf.losses.mean_squared_error(labels_, predictions)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer


def build_accuracy(predictions, labels_):
    """
    Create accuracy
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size,
                            learning_rate, keep_prob, train_x, val_x, train_y, val_y):
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        n_batches = len(train_x) // batch_size
        for e in range(epochs):
            state = sess.run(initial_state)

            train_acc = []
            for ii, (x, y) in enumerate(utl.get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob_: keep_prob,
                        initial_state: state}
                loss_, state, _, batch_acc = sess.run([loss, final_state, optimizer, accuracy], feed_dict=feed)
                train_acc.append(batch_acc)

                if (ii + 1) % batch_size == 0:

                    val_acc = []
                    val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    for xx, yy in utl.get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: xx,
                                labels_: yy[:, None],
                                keep_prob_: 1,
                                initial_state: val_state}
                        val_batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(val_batch_acc)

                    print("Epoch: {}/{}...".format(e + 1, epochs),
                          "Batch: {}/{}...".format((ii + 1)//batch_size, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))

        saver.save(sess, "checkpoints/sentiment.ckpt")



# Define Inputs and Hyperparameters
lstm_sizes = [128, 64]
vocab_size = len(vocab_to_int) + 1 #add one for padding
embed_size = 300
epochs = 50
batch_size = 256
learning_rate = 0.1
keep_prob = 0.5

with tf.Graph().as_default():
    build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size,
                            learning_rate, keep_prob, train_x, val_x, train_y, val_y)




# Test：

def test_network(model_dir, batch_size, test_x, test_y):
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)

    saver = tf.train.Saver()

    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        test_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(utl.get_batches(test_x, test_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob_: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test Accuracy: {:.3f}".format(np.mean(test_acc)))

with tf.Graph().as_default():
    test_network('checkpoints', batch_size, test_x, test_y)