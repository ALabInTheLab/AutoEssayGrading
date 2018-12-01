import tensorflow as tf
import numpy as np
import sys

from helper import readData

def last_relevant(seq, length):
    batch_size = tf.shape(seq)[0]
    max_length = int(seq.get_shape()[1])
    input_size = int(seq.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(seq, [-1, input_size])
    return tf.gather(flat, index)

def main():
    articles, labels = readData(0)

    print(articles.shape)
    print(labels.shape)
    sys.exit()

    input_text = tf.placeholder(tf.int32, shape=[None, sequence_length], name='input_text')
    input_y = tf.placeholder(tf.float32, shape=[None, num_classes], name='input_y')
    dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout_keep_prob)
    all_outputs, _ = tf.nn.dynamic_rnn(cell=cell,
                                        inputs=input_text,
                                        sequence_length=text_length,
                                        dtype=tf.float32)
    h_outputs = last_relevant(all_outputs, text_length)

    # Final scores and predictions
    W = tf.get_variable("W", shape=[hidden_size, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    logits = tf.nn.xw_plus_b(h_outputs, W, b, name="logits")
    predictions = tf.argmax(logits, 1, name="predictions")

    # Calculate mean cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=input_y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

    # Accuracy
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")

    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(rnn.loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run([train_op, loss, accuracy], feed_dict)


if __name__ == '__main__':
    main()