import json
from typing import Iterator, Optional, List

import numpy as np
import tensor_annotations.tensorflow as ttf
import tensorflow as tf
from tensor_annotations import axes
from tensorflow.keras import Model

from model import Anchors, Batch, Channels, Width

"""
Dataset format:
    ndjson format
    [setence, [[tag, [begin, in_tag]], ...]]
    ...
    ez)
    ["책머리에 ", [["NNG", [0, 3]], ["JKB", [3, 1]]]]
"""

class Vocab:
    def __init__(self, iterator: Iterator[str], special_tokens=[], max_len=None):
        self._itos = special_tokens + list(set(iterator))
        self._stoi = {s: i for i, s in enumerate(self._itos)}
    def itos(self, i: int) -> str:
        return self._itos[i]
    def stoi(self, s: str) -> int:
        return self._stoi.get(s)
    def __len__(self) -> int:
        return len(self._itos)


def load_dataset(
    dataset_pattern,
    batch_size=1,
    buffer=30,
    max_text_len=256,
    anchors=[2,4,6],
    text_vocab: Optional[Vocab] = None,
    label_vocab: Optional[Vocab] = None,
):
    if text_vocab is None:
        text_vocab = Vocab(
                (c for line in open(dataset_pattern, "r") for c in json.loads(line)[0]), 
                special_tokens=['<pad>'], 
                max_len=2000)
    if label_vocab is None:
        label_vocab = Vocab((tag[0] for line in open(dataset_pattern, "r") for tag in json.loads(line)[1]))
    print('text vocab size:', len(text_vocab))
    print('label vocab size:', len(label_vocab))
    def parse_line(batches: Iterator[str], max_text_len: int, anchors: List[int]) -> (ttf.Tensor3[Batch, Width, Channels], ttf.Tensor4[Batch, Width, Anchors, Channels]):
        uniform_distribution = np.full(len(label_vocab), 1.0 / len(label_vocab))
        deta = 0.01
        batches = batches.numpy()
        xs = np.zeros((batches.shape[0], max_text_len, len(text_vocab)), dtype=np.float32)
        ys = np.zeros((batches.shape[0], max_text_len, len(anchors), 3 + len(label_vocab)), dtype=np.float32)
        #sents, tags, xs, ws = [], [], [], []
        for i, line in enumerate(batches):
            sent, tag_and_xw_bboxes = json.loads(line)
            sent = [text_vocab.stoi(s) for s in sent[:max_text_len]] + [text_vocab.stoi('<pad>') for _ in range(max_text_len - len(sent))]
            sent = np.eye(len(text_vocab))[sent]
            xs[i, :, :] = sent
            for j, (tag, xw_bbox) in enumerate(tag_and_xw_bboxes):
                label = label_vocab.stoi(tag)
                label = np.eye(len(label_vocab))[label]
                label = label * (1 - deta) + deta * uniform_distribution
                ys[i, j, :, :] = np.concatenate((xw_bbox, [1.0], label), axis=-1)
        return (xs, ys)
    def set_shape(xs, ys):
        xs.set_shape((batch_size, max_text_len, len(text_vocab)))
        ys.set_shape((batch_size, max_text_len, len(anchors), len(label_vocab) + 3))
        return (xs, ys)
    x = tf.data.TextLineDataset(dataset_pattern)
    x = x.shuffle(10)
    x = x.batch(batch_size)
    x = x.map(
        lambda x: tf.py_function(
            parse_line, [x, max_text_len, anchors], [tf.float32, tf.float32]
        )
    )
    x = x.map(set_shape)
    x.prefetch(buffer)
    return x, text_vocab, label_vocab
