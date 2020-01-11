# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng

'''
文本2向量转换器
1. 基于静态词典的one-hot表示
2. 基于静态词向量的稠密表示
3. 基于bert_api的动态语言模型表示
'''
import os
import numpy as np
import collections


class Text2Vector(object):
    def __init__(self, w2v_file="w2v_ch_wiki/words.vector", binary=True, normalized=False, by_dict=False, input_len=None,
                 weight_func=None, seg_func=None):
        self.bc = None
        self.weight_func = weight_func
        self.unit_type = "char"
        self.by_dict = by_dict
        self.seg_func = seg_func
        if w2v_file == "bert_api":
            from bert_serving.client import BertClient
            self.name = "bert"
            self.max_len = 32
            self.vec_dim = 768
            self.output_shape = [None, self.max_len, self.vec_dim]
            self.bc = BertClient(ip='10.11.2.10', port=8001, port_out=8002, show_server_config=False, output_fmt="list")
        elif self.by_dict:
            self.name = "dict"
            with open(w2v_file, "r", encoding="utf-8") as fp:
                corpus_dict = [w.strip().replace(" ", "") for w in fp.readlines() if len(w.strip().replace(" ", "")) > 0]
            if "__unk__" not in corpus_dict:
                corpus_dict = ["__unk__"] + corpus_dict
            if "__pad__" not in corpus_dict:
                corpus_dict = ["__pad__"] + corpus_dict
            self.vec_dim = len(corpus_dict)
            self.word2ix = {w: i for i, w in enumerate(corpus_dict)}
            self.input_len, self.max_len = input_len, input_len
            self.output_shape = [None, input_len]
            self.unk_ix = self.word2ix.get("__unk__")
            self.pad_ix = self.word2ix.get("__pad__")
            # self.oov_ix = corpus_dict.insert("__unk__")
            # self.pad_vec = self.word_vec("__pad__")
            # self.oov_vec = self.word_vec("__oov__")
        else:
            self.name = "w2v"
            from gensim import models
            import platform
            self.w2v_path = w2v_file
            if platform.system() == 'Darwin':
                self.nlp_path = "/Users/fanzfeng/Data/nlpSource/"
                self.w2v_path = os.path.join(self.nlp_path, w2v_file)
            self.unit_type = "word"
            self.max_len = input_len
            try:
                self.w2v = models.Word2Vec.load(self.w2v_path).wv
            except:
                self.w2v = models.KeyedVectors.load_word2vec_format(self.w2v_path, binary=binary)
            self.voca = self.w2v.index2entity
            self.vector_size = len(self.w2v.get_vector(self.voca[0]))
            self.vec_dim = self.w2v.vector_size
            self.output_shape = [None, input_len, self.vec_dim]
            self.norm = normalized
            self.pad_vec = self.word_vec("__pad__", na_rand=True)
            if normalized:
                self.w2v.init_sims(replace=True) # save memory

    def word_vec(self, word, na_rand=False):
        if self.bc is not None:
            return None
        elif self.by_dict:
            vec_tmp = [0]*self.vec_dim
            ix = self.word2ix.get(word, self.unk_ix)
            vec_tmp[ix] += 1
            return vec_tmp
        if not na_rand:
            return self.w2v.get_vector(word)
        elif word in self.voca:
            return self.w2v.get_vector(word)
        else:
            v_range = 10 if not self.norm else 1
            random_state = np.random.RandomState(seed=(hash(word) % (2 ** 32 - 1)))
            return random_state.uniform(low=-v_range, high=v_range, size=(self.vector_size,))

    def text2vec(self, text, avg=True, output_array=True):
        xlen = len(text)
        if self.bc is not None:
            if isinstance(text, str):
                last_ix = xlen+2
                vec = self.bc.encode([text])
                if avg:
                    return np.array(vec[0])[0:min(last_ix, self.max_len), :].mean(axis=0)
                elif output_array:
                    return np.array(vec[0][0])
                else:
                    return vec[0]
            elif isinstance(text, (tuple, list)):
                res_list = self.bc.encode(text)
                if avg:
                    res = [np.array(v) for v in res_list]
                    return [res[j][0:min(len(text[j])+2, self.max_len), :].mean(axis=0).tolist() for j in range(xlen)]
                elif output_array:
                    return np.array([r[0] for r in res_list])
                else:
                    return res_list
            else:
                raise ValueError
        if isinstance(text, str) and xlen > 0 and self.seg_func:
            text = self.seg_func(text)
            xlen = len(text)
        if self.by_dict:
            if isinstance(self.input_len, int) and self.input_len > 0:
                vec_list = [self.word2ix.get(text[j], self.unk_ix) if j < xlen else self.pad_ix
                            for j in range(self.input_len)]
                return np.array(vec_list) if output_array else vec_list
            else:
                return [self.word2ix.get(text[j], self.unk_ix) for j in range(xlen)]

        d = collections.OrderedDict()
        vec_list = []
        for w in text:
            if w not in d:
                d[w] = 1/xlen
                vec_list += [self.word_vec(word=w, na_rand=True)]
            else:
                d[w] += 1/xlen
        mx = np.array(vec_list)
        if avg:
            weight = np.array(self.weight_func(text) if self.weight_func is not None else [v for k, v in d.items()])
            mx_avg = np.average(mx, axis=0, weights=weight/weight.sum())
        if not output_array:
            if avg:
                return mx_avg.tolist()
            elif self.max_len is not None:
                return [vec_list[j] if j < xlen else self.pad_vec for j in range(self.max_len)]
            return vec_list
        else:
            return mx_avg if avg else mx


if __name__ == "__main__":
    pass