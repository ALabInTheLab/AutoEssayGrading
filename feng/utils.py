import re
import string
from collections import Counter
import numpy as np

def zero_pad_messages(messages, seq_len):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param seq_ken: Input int, maximum sequence input length
    :return: numpy array.  The encoded labels
    """
    messages_padded = np.zeros((len(messages), seq_len), dtype=int)
    for i, row in enumerate(messages):
        messages_padded[i, -len(row):] = np.array(row)[:seq_len]

    return np.array(messages_padded)

def train_val_test_split(messages, labels, split_frac, random_seed=None):
    """
    Zero Pad input messages
    :param messages: Input list of encoded messages
    :param labels: Input list of encoded labels
    :param split_frac: Input float, training split percentage
    :return: tuple of arrays train_x, val_x, test_x, train_y, val_y, test_y
    """
    # make sure that number of messages and labels allign
    assert len(messages) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(messages))
    messages_shuf = np.array(messages)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(messages_shuf)*split_frac)
    train_x, val_x = messages_shuf[:split_idx], messages_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y
    
def get_batches(x, y, batch_size=100, needall=False):
    """
    Batch Generator for Training
    :param x: Input array of x data
    :param y: Input array of y data
    :param batch_size: Input int, size of batch
    :return: generator that returns a tuple of our x batch and y batch
    """
    n_batches = len(x)//batch_size
    residue = len(x) % batch_size
    has_residue = False
    if residue > 0:
        has_residue = True

    # x, y = xin[:n_batches*batch_size], yin[:n_batches*batch_size]
    lengths = [essay.shape[0] for essay in x]
    for ii in range(0, len(x), batch_size):
        # assert ii <= n_batches * batch_size
        if ii >= n_batches * batch_size:
            if has_residue and needall:
                output = batch_padding(x[ii:ii+batch_size], y[ii:ii+batch_size], lengths[ii:ii+batch_size], batch_size)
                # automatically cut off if ii+batch_size is beyond the range
                # yield output[0], output[1], output[2], residue
                yield output + (residue,)
            else:
                break
        else:
            yield x[ii:ii+batch_size], y[ii:ii+batch_size], lengths[ii:ii+batch_size], batch_size

    # if has_residue and needall:
    #     output = batch_padding(x[n_batches * batch_size : len(x)], y[n_batches*batch_size:len(x)], lengths[n_batches*batch_size:len(x)], batch_size)
    #     yield output[0], output[1], output[2], residue


def batch_padding(X_test, Y_test, seqlen_origin, batch_size):
    test_size = Y_test.shape[0]
    # y = np.zeros(batch_size, dtype=int)
    seqlen = np.zeros(batch_size, dtype=int)
    # y[:test_size] = Y_test
    seqlen[:test_size] = seqlen_origin

    to_append = [np.zeros_like(X_test[0]) for _ in range(test_size, batch_size)]
    # X_test.append(to_append)
    X_test = X_test + to_append
    # for _ in range(test_size, batch_size):
        # X_test.append(to_append)

    # to_append = np.zeros_like(Y_test[0])
    # to_append = np.ones((batch_size - test_size))[:, None] * to_append[None, :]
    to_append = one_hot(np.ones(batch_size - test_size, dtype=int), Y_test.shape[1])
    Y_test = np.append(Y_test[:, :], to_append, axis=0)
    # for _ in range(test_size, batch_size):
    #     np.append(Y_test, to_append)

    return X_test, Y_test, seqlen



# Utils:
def one_hot(Y, n_classes):
    batch_size = Y.shape[0]
    output = np.zeros((batch_size, n_classes), dtype=int)
    # for idx in range(batch_size):
    #     output[idx, Y[idx]-1] = 1

    output[np.arange(batch_size), Y.astype(int) - 1] = 1
    # output[np.arange(batch_size), Y.astype(int)] = 1
    # return output
    return output


def one_hot_reverse(Y):
    output = np.argwhere(Y == 1)[:, 1] + 1
    # output = np.argwhere(Y == 1)[:, 1]

    return output


def batch_zero_pad(x_batch):
    # padding zeros along batch diretion
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


def hist_freq(y, num_classes):
    freq = np.zeros(num_classes, dtype=float)
    weight = np.zeros(num_classes, dtype=float)
    for c in y:
        freq[int(c)-1] += 1

    weight[freq>0] = 1/freq[freq>0]

    return freq, weight



