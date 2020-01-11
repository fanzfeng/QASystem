# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 22/11/2019
# @Author  : fanzfeng
'''
todo:
1. 提供tfidf的稀疏表示形式
'''

import os
import numpy as np
import pandas as pd
import collections
import sys
root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from nlp_utils.nlp_base import NgramsSeg as NGrams


class TfIdf(object):
    def __init__(self, corpus_files=[], sep="\t", text_col_ix=0, ngrams=1, seg_func=None, corpus_distinct=True,
                 base_path=None, output_model="idf", sparse=False, input_len=None):
        self.sparse = sparse
        self.output_model = output_model
        self.pad_str = "__pad__"
        self.seg_func = seg_func
        self.idf_dict = {}
        self.unit_freq = {}
        self.ngrams = ngrams

        corpus = []
        if isinstance(base_path, str) and os.path.exists(base_path):
            corpus_files = [os.path.join(base_path, f) for f in corpus_files]
        for f in corpus_files:
            if os.path.exists(f):
                df = pd.read_csv(f, encoding="utf-8", sep=sep)
                cols = list(df.columns)
                corpus += [s for s in df[cols[text_col_ix]].tolist() if isinstance(s, str)]
        if corpus_distinct and len(corpus) > 0:
            corpus = set(corpus)

        self.corpus_size = len(corpus)
        if self.corpus_size > 0:
            if ngrams >= 1 and isinstance(ngrams, int):
                kernel = NGrams(ngrams=ngrams, out_list=True)
                self.seg_func = kernel.text_seg
            self.corpus_seg = [self.seg_func(c) for c in corpus]

        self.vocab = []
        for s in self.corpus_seg:
            for u in s:
                if u not in self.vocab:
                    self.vocab.append(u)
        self.input_len = input_len
        self.vec_dim = len(self.vocab)
        self.output_shape = (None, input_len)
        self.idf_default = np.log10(self.corpus_size)
        self.idf()

    def idf(self):
        if self.corpus_seg:
            for u in self.vocab:
                cnt = sum(1 if u in s else 0 for s in self.corpus_seg)
                self.idf_dict.update({u: np.log10(self.corpus_size/(1+cnt))})

    def idf_get(self, u):
        return self.idf_dict.get(u, self.idf_default)

    def text2vec(self, text):
        if isinstance(text, str) and len(text) > 0:
            len_0 = len(text)
            if isinstance(self.input_len, int) and self.ngrams > 0:
                text = list(text)
                text = text[0:self.input_len] if len_0 >= self.input_len else text+[self.pad_str]*(self.input_len-len_0)
                if self.ngrams == 2:
                    text += [self.pad_str]
                elif self.ngrams == 3:
                    text = [self.pad_str] + text + [self.pad_str]
            text_seg = self.seg_func(text)
        elif isinstance(text, (list, tuple)) and len(text) > 0:
            text_seg = text
        if text_seg:
            nlen = len(text_seg)
            unit2tf = collections.OrderedDict()
            for u in text_seg:
                if u in unit2tf:
                    unit2tf[u] += 1/nlen
                else:
                    unit2tf[u] = 1/nlen
            u_set = self.vocab if self.sparse else unit2tf
            if self.output_model == "idf":
                return [self.idf_get(u) for u in u_set]
            elif self.output_model == "tf":
                return [unit2tf.get(u, 0) for u in u_set]
            elif self.output_model == "tfidf":
                return [unit2tf.get(u, 0)*self.idf_get(u) for u in u_set]


if __name__ == "__main__":
    pass
