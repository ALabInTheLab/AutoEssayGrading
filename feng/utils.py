import re
import string
from collections import Counter
import numpy as np

def preprocess_ST_message(text):
    """
    Preprocesses raw message data for analysis
    :param text: String. ST Message
    :return: List of Strings.  List of processed text tokes
    """
    # Define ST Regex Patters
    REGEX_PRICE_SIGN = re.compile(r'\$(?!\d*\.?\d+%)\d*\.?\d+|(?!\d*\.?\d+%)\d*\.?\d+\$')
    REGEX_PRICE_NOSIGN = re.compile(r'(?!\d*\.?\d+%)(?!\d*\.?\d+k)\d*\.?\d+')
    REGEX_TICKER = re.compile('\$[a-zA-Z]+')
    REGEX_USER = re.compile('\@\w+')
    REGEX_LINK = re.compile('https?:\/\/[^\s]+')
    REGEX_HTML_ENTITY = re.compile('\&\w+')
    REGEX_NON_ACSII = re.compile('[^\x00-\x7f]')
    REGEX_PUNCTUATION = re.compile('[%s]' % re.escape(string.punctuation.replace('<', '')).replace('>', ''))
    REGEX_NUMBER = re.compile(r'[-+]?[0-9]+')

    text = text.lower()

    # Replace ST "entitites" with a unique token
    text = re.sub(REGEX_TICKER, ' <TICKER> ', text)
    text = re.sub(REGEX_USER, ' <USER> ', text)
    text = re.sub(REGEX_LINK, ' <LINK> ', text)
    text = re.sub(REGEX_PRICE_SIGN, ' <PRICE> ', text)
    text = re.sub(REGEX_PRICE_NOSIGN, ' <NUMBER> ', text)
    text = re.sub(REGEX_NUMBER, ' <NUMBER> ', text)
    # Remove extraneous text data
    text = re.sub(REGEX_HTML_ENTITY, "", text)
    text = re.sub(REGEX_NON_ACSII, "", text)
    text = re.sub(REGEX_PUNCTUATION, "", text)
    # Tokenize and remove < and > that are not in special tokens
    words = " ".join(token.replace("<", "").replace(">", "")
                     if token not in ['<TICKER>', '<USER>', '<LINK>', '<PRICE>', '<NUMBER>']
                     else token
                     for token
                     in text.split())

    return words

def create_lookup_tables(words):
    """
    Create lookup tables for vocabulary
    :param words: Input list of words
    :return: A tuple of dicts.  The first dict maps a vocab word to and integeter
             The second maps an integer back to to the vocab word
    """
    word_counts = Counter(words)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab

def encode_ST_messages(messages, vocab_to_int):
    """
    Encode ST Sentiment Labels
    :param messages: list of list of strings. List of message tokens
    :param vocab_to_int: mapping of vocab to idx
    :return: list of ints. Lists of encoded messages
    """
    messages_encoded = []
    for message in messages:
        messages_encoded.append([vocab_to_int[word] for word in message.split()])

    return np.array(messages_encoded)

def encode_ST_labels(labels):
    """
    Encode ST Sentiment Labels
    :param labels: Input list of labels
    :return: numpy array.  The encoded labels
    """
    return np.array([1 if sentiment == 'bullish' else 0 for sentiment in labels])

def drop_empty_messages(messages, labels):
    """
    Drop messages that are left empty after preprocessing
    :param messages: list of encoded messages
    :return: tuple of arrays. First array is non-empty messages, second array is non-empty labels
    """
    non_zero_idx = [ii for ii, message in enumerate(messages) if len(message) != 0]
    messages_non_zero = np.array([messages[ii] for ii in non_zero_idx])
    labels_non_zero = np.array([labels[ii] for ii in non_zero_idx])
    return messages_non_zero, labels_non_zero

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
                yield output[0], output[1], output[2], residue
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



