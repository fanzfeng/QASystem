# -*- coding: utf-8 -*-
# version=3.6.4
# @Date  : 22/11/2019
# @Author  : fanzfeng

import os
import pandas as pd
import sys
root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from nlp_utils.nlp_base import NgramsSeg


class Ngrams(object):
    def __init__(self, corpus_files=[], sep="\t", text_col_ix=0, ngrams=1, seg_func=None, base_path=None,
                 output_type="list", vocab_file=None, input_len=None, output_align=False):
        self.output_type = output_type
        self.output_align = output_align
        self.seg_func = seg_func
        self.name = "dict"
        self.ngrams = ngrams
        self.pad_str = "__pad__"
        self.unk_str = "__unk__"
        if self.seg_func is None and ngrams >= 1 and isinstance(ngrams, int):
            kernel = NgramsSeg(ngrams=ngrams, out_list=True)
            self.seg_func = kernel.text_seg
        if isinstance(vocab_file, str) and os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as fp:
                self.corpus_dict = [w.strip().replace(" ", "") for w in fp.readlines()
                                    if len(w.strip().replace(" ", "")) > 0]
        else:
            self.corpus_dict, corpus = [], []
            if isinstance(base_path, str) and os.path.exists(base_path):
                corpus_files = [os.path.join(base_path, f) for f in corpus_files]
            for f in corpus_files:
                if os.path.exists(f):
                    df = pd.read_csv(f, encoding="utf-8", sep=sep)
                    cols = list(df.columns)
                    corpus += [s for s in df[cols[text_col_ix]].tolist() if isinstance(s, str)]
            if len(corpus) > 0:
                assert self.seg_func is not None
                for s in set(corpus):
                    for u in self.seg_func(s):
                        if u not in self.corpus_dict:
                            self.corpus_dict.append(u)
        if self.unk_str not in self.corpus_dict:
            self.corpus_dict = [self.unk_str] + self.corpus_dict
        if self.pad_str not in self.corpus_dict:
            self.corpus_dict = [self.pad_str] + self.corpus_dict
        self.vec_dim = len(self.corpus_dict)
        self.output_shape = [None, input_len]
        self.word2ix = {w: i for i, w in enumerate(self.corpus_dict)}
        self.input_len, self.max_len = input_len, input_len
        self.unk_ix = self.word2ix.get(self.unk_str)
        self.pad_ix = self.word2ix.get(self.pad_str)

    def text2vec(self, text):
        if not isinstance(text, str) or len(text) < 1:
            return []
        len_0 = len(text)
        if self.output_align and isinstance(self.input_len, int) and self.ngrams > 0:
            text = list(text)
            text = text[0:self.input_len] if len_0 >= self.input_len else text + [self.pad_str] * (self.input_len - len_0)
            if self.ngrams == 2:
                text += [self.pad_str]
            elif self.ngrams == 3:
                text = [self.pad_str] + text + [self.pad_str]
        text = self.seg_func(text)
        if isinstance(self.input_len, int) and self.input_len > 0:
            vec_list = [
                self.word2ix.get(text[j], self.unk_ix) if j < len(text) else self.pad_ix for j in range(self.input_len)]
        else:
            vec_list = [
                self.word2ix.get(text[j], self.unk_ix) for j in range(len(text))]
        return vec_list


if __name__ == "__main__":
    pass