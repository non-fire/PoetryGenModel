import os
import json
import re
import torch
import random
import collections
import _pickle as p
import numpy as np
import torch.nn.functional as F

max_words = 30

def dataHandler(batch_size, num_steps):
    def load(fname):
        res = []
        data = json.loads(open(fname, encoding='utf-8').read())
        for poem in data:
            pdata=""
            p = poem.get("paragraphs")
            for sen in p:
                pdata+=sen
            pdata, _ = re.subn("（.*）", "", pdata)
            pdata, _ = re.subn("（.*）", "", pdata)
            pdata, _ = re.subn("{.*}", "", pdata)
            pdata, _ = re.subn("《.*》", "", pdata)
            pdata, _ = re.subn("《.*》", "", pdata)
            pdata, _ = re.subn("[\]\[]", "", pdata)
            r = ""
            for s in pdata:
                if s not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']:
                    r += s
            r, _ = re.subn("。。", "。", r)
            if r != "":
                res.append(r)
            res.append(r)
        return res

    def vocab_gen(data):
        '''

        counter = count_corpus(tokens)
        token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        token_freqs=token_freqs[:max_words-3]
        words=[token[0] for token in token_freqs]
        '''
        tokens = [list(line) for line in data]
        tokens = [token for line in tokens for token in line]
        vocab = {}
        for token in tokens:
            vocab.setdefault(token, len(vocab))
        vocab['<EOP>'] = len(vocab)
        vocab['<START>'] = len(vocab)
        return vocab

    def corpus_gen(data, vocab):
        corpus=[[vocab[w] for w in line] for line in data]
        return corpus

    def pad_sequences(sequences,
                      maxlen=None,
                      dtype='int32',
                      padding='pre',
                      truncating='pre',
                      value=0.):
        """
        code from keras
        Pads each sequence to the same length (length of the longest sequence).
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen.
        Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
        Arguments:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than
                maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        Returns:
            x: numpy array with dimensions (number_of_sequences, maxlen)
        Raises:
            ValueError: in case of invalid values for `truncating` or `padding`,
                or in case of invalid shape for a `sequences` entry.
        """
        if not hasattr(sequences, '__len__'):
            raise ValueError('`sequences` must be iterable.')
        lengths = []
        for x in sequences:
            if not hasattr(x, '__len__'):
                raise ValueError('`sequences` must be a list of iterables. '
                                 'Found non-iterable: ' + str(x))
            lengths.append(len(x))

        num_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:  # pylint: disable=g-explicit-length-test
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if not len(s):  # pylint: disable=g-explicit-length-test
                continue  # empty list/array was found
            if truncating == 'pre':
                trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError(
                    'Shape of sample %s of sequence at position %s is different from '
                    'expected shape %s'
                    % (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x

    data = []
    src = './autodl-tmp/json/'
    for filename in os.listdir(src):
        if filename.startswith("poet.tang"):
            data.extend(load(src + filename))
    #for i in range(0, len(data)):
        #data[i] = ['<START>'] + list(data[i]) + ['<EOP>']

    vocab = vocab_gen(data)
    corpus = corpus_gen(data, vocab)
    write_file = open('wordDic', 'wb')
    p.dump(vocab, write_file)
    pad_data = pad_sequences(corpus,
                             max_words,
                             padding='pre',
                             truncating='post',
                             value=len(vocab) - 1)
    return pad_data, vocab