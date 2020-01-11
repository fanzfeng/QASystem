# -*- coding: utf-8 -*-
# @Date  : 2019/3/12
# @Author  : fanzfeng
'''
similarities.MatrixSimilarity is only appropriate when the whole set of vectors fits into memory.
similarities.Similarity class operates in fixed memory, by splitting the index across multiple files on disk
'''

import os
import time
import logging
import platform
import numpy as np
from gensim.summarization.bm25 import BM25
from gensim.models import TfidfModel, KeyedVectors, Word2Vec
from gensim import corpora
from gensim import similarities
from gensim import matutils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)

# log_path = "/Users/fanzfeng/project_code/feature-nlp/za_nlp/TextSim/log/log.faq"
# log_file = logging.FileHandler(log_path)
# log_file.setLevel(logging.DEBUG)
# log_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# logger.addHandler(log_file)


def cos_sim(vec1, vec2):
    # 存在计算精度的问题
    npvec1, npvec2 = np.array(vec1), np.array(vec2)
    cos = npvec1.dot(npvec2)/(np.sqrt((npvec1**2).sum()) * np.sqrt((npvec2**2).sum()))
    return cos if cos <= 1.0 else 1.0


def levenshtein_sim(sentence1, sentence2, sim=True):
    first, second = sentence1, sentence2
    sentence1_len, sentence2_len = len(first), len(second)
    maxlen = max(sentence1_len, sentence2_len)
    if sentence1_len > sentence2_len:
        first, second = second, first

    distances = range(len(first) + 1)# 短串+1
    for index2, char2 in enumerate(second):# 长字符串
        new_distances = [index2 + 1] #第几个字符串
        for index1, char1 in enumerate(first): # 短字符串
            if char1 == char2:
                new_distances.append(distances[index1]) #distances[ix]=ix
            else:
                min_ix = min((distances[index1], distances[index1+1], new_distances[-1]))
                new_distances.append(1+min_ix)
        distances = new_distances
    levenshtein = distances[-1]
    return float((maxlen - levenshtein) / maxlen) if sim else levenshtein


def jaccard_sim(x1, x2):
    s1, s2 = set(x1), set(x2)
    jac = len(s1 & s2)/len(s1 | s2)
    return jac


class BM25(object):
    def __init__(self, texts):
        assert isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], list)
        time_init = time.time()
        # self.dct = corpora.Dictionary(texts)
        # self.corpus_ = [self.dct.doc2bow(text) for text in texts]
        self.model = BM25(texts)
        self.doc_num = len(texts)
        self.avg_idf = sum(map(lambda k: float(self.model.idf[k]), self.model.idf.keys())) / len(self.model.idf.keys())
        logger.warning("Build BM25 model use time %s", time.time()-time_init)

    def sim_search(self, words_cut):
        scores_list = self.model.get_scores(words_cut, self.avg_idf)
        res = zip([k for k in range(self.doc_num)], scores_list)
        return sorted(res, key=lambda p: p[1], reverse=True)


class TfIdf(object):
    '''
    SparseMatrixSimilarity contain function matutils.corpus2csc
    '''
    def __init__(self, texts):
        time_init = time.time()
        self.dct = corpora.Dictionary(texts)
        # dct.save('/tmp/deerwester.dict')
        self.dict_size = len(self.dct.token2id)
        self.doc_num = len(texts)
        self.corpus_ = [self.dct.doc2bow(corpus) for corpus in texts]
        self.model = TfidfModel(self.corpus_)
        self.index = similarities.SparseMatrixSimilarity(self.model[self.corpus_], num_features=self.dict_size)
        # index.save('/tmp/deerwester.index')
        # index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')
        logger.warning("Build TFIDF model use time %s", time.time() - time_init)

    def text2vec(self, text, norm=False):
        vec = self.model[self.dct.doc2bow(text)]
        if norm:
            matutils.unitvec(vec, norm='l2')
        return vec

    def sim_search(self, text):
        vec = self.text2vec(text)
        sims = self.index[self.model[vec]]
        return sorted(list(enumerate(sims)), key=lambda p: p[1], reverse=True)

    def sim_rank(self, query, doc_corpus, query_norm=True, doc_norm=True, res_sort=True):
        assert isinstance(doc_corpus, (tuple, list)) and len(doc_corpus) > 0
        if not isinstance(doc_corpus[0], (tuple, list)):
            doc_corpus = [doc_corpus]
        corpus_vec = [self.text2vec(c, norm=doc_norm) for c in doc_corpus]
        doc_index = matutils.corpus2csc(corpus_vec, num_terms=self.dict_size, dtype=np.float32).T.tocsr()
        query_vec = self.text2vec(query, norm=query_norm)
        query_index = matutils.corpus2csc([query_vec], num_terms=self.dict_size, dtype=doc_index.dtype).tocsr()
        sim_array = doc_index * query_index
        sims = sim_array.toarray().T[0].tolist()
        # sims = doc_index[query_vec]
        return sorted(list(enumerate(sims)), key=lambda p: p[1], reverse=True) if res_sort else sims


class W2V(object):
    def __init__(self, w2v_file="w2v_ch_wiki/words.vector", binary=True, normalized=False):
        time_init = time.time()
        if platform.system() == 'Darwin':
            self.nlp_path = "/Users/fanzfeng/Data/nlpSource/"
        else:
            self.nlp_path = "E:\\nlp_source"
        if os.path.exists(w2v_file):
            self.w2v_path = w2v_file
        else:
            self.w2v_path = os.path.join(self.nlp_path, w2v_file)
        try:
            self.w2v = Word2Vec.load(self.w2v_path).wv
        except:
            self.w2v = KeyedVectors.load_word2vec_format(self.w2v_path, binary=binary)
        self.voca = self.w2v.index2entity
        self.vector_size = len(self.w2v.get_vector(self.voca[0]))
        self.norm = normalized
        if normalized:
            self.w2v.init_sims(replace=True) # save memory
        logger.warning("Build W2v model use time %s", time.time() - time_init)

    def word_vec(self, word, na_rand=False):
        if not na_rand:
            return self.w2v.get_vector(word)
        elif word in self.voca:
            return self.w2v.get_vector(word)
        else:
            v_range = 10 if not self.norm else 1
            random_state = np.random.RandomState(seed=(hash(word) % (2 ** 32 - 1)))
            return random_state.uniform(low=-v_range, high=v_range, size=(self.vector_size,))

    def text2vec(self, text, avg=True):
        vec_list = []
        for w in text:
            vec_list += [self.word_vec(word=w, na_rand=True)]
        mx = np.array(vec_list)
        return mx.mean(axis=0) if avg else mx

    def neighbours(self, word, size=10):
        if word in self.voca:
            return self.w2v.similar_by_word(word, topn=size, restrict_vocab=None)

    def word_sim(self, w1, w2, new_word_cal=False):
        if not new_word_cal:
            return self.w2v.similarity(w1, w2)
        else:
            v1 = self.word_vec(w1, na_rand=True)
            v2 = self.word_vec(w2, na_rand=True)
            return cos_sim(v1, v2)

    def wmd_distance(self, sent1, sent2):
        if isinstance(sent1, list) and isinstance(sent2, list):
            return self.w2v.wmdistance(sent1, sent2)
        else:
            raise ValueError("input should be list")


if __name__ == "__main__":
    pass
    # word2vec = W2V(w2v_file="w2v_ch_wiki/words_vector.vec", binary=True)
    # word2vec.word_vec("中国")
    # print(word2vec.wmd_distance(sent1="你 哪个 地址".split(), sent2="把 位置 发我 吧".split()))
