# -*- coding: utf-8 -*-
# @Date  : 2019/10/24
# @Author  : fanzfeng
'''
语义检索：向量表示+向量检索
    annoy 向量近邻检索：https://github.com/spotify/annoy

'''

import os
import time
import logging
import pandas as pd
from annoy import AnnoyIndex
import sys
root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from text2vector.vector_model import Text2Vector

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)


class SearchEngine(object):
    def __init__(self, query2rid_file, res_col="answer", query_col="question", index_col=None, cache_file=None,
                 data_sep="\t", vec_model=["bert", "w2v", "dict"][1], seg_func=None, weight_func=None,
                 model_file=None):
        self.query2rid_file = query2rid_file
        self.data_sep = data_sep
        self.index_col = index_col
        self.res_col = res_col
        self.query_col = query_col
        self.vec_model = vec_model

        self.weight_func = weight_func
        self.seg_func = seg_func

        self.res_flatten = False
        self.res_index_col = "index_text"
        self.res_origin_col = "query_text"
        self.res_sim_col = "score"
        self.res_id_col = "qid"
        self.res_index_id = "query_ix"

        if vec_model == "w2v" and isinstance(model_file, str):
            self.vector_gen = Text2Vector(w2v_file=model_file, by_dict=False, input_len=None, seg_func=self.seg_func,
                                          weight_func=self.weight_func)
        elif vec_model == "bert":
            self.vector_gen = Text2Vector(w2v_file="bert_api")
        else:
            raise ValueError("vec_model is invalid...")
        self.vec_dim = self.vector_gen.vec_dim

        self.query2res = {} # query_text2std_id(answer_id)
        self.index_data = {} # query_text2id_list
        self.process_data()

        self.ix2doc = {}  # qid2query_text
        self.engine = AnnoyIndex(self.vec_dim, 'angular')
        self.tree_rebuild = True
        self.cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            logger.info(" Loading annoy engine...")
            try:
                self.engine.load(cache_file)
                logger.info(" Annoy engine with {} items ready".format(self.engine.get_n_items()))
                self.tree_rebuild = False
            except Exception as e:
                logger.warning(" Annoy vector error {}".format(e))
        self.build_index()

    def check_index(self, data):
        if self.index_col:
            group_col = self.index_col
        else:
            group_col = self.query_col
        if isinstance(data, pd.core.frame.DataFrame):
            if sum(c not in data.columns for c in [group_col, self.res_col]) <= 0:
                cnt = data.groupby(group_col).apply(lambda r: r[self.res_col].unique().shape[0])
                if (cnt > 1).sum() > 0:
                    error_index = set(cnt[cnt > 1].index)
                    logger.warning(" Contradicts exist in index data: {}".format(error_index))

    def process_data(self):
        query2rid_data = None
        if isinstance(self.query2rid_file, str):
            logger.info(" Read index from file {}".format(self.query2rid_file))
            assert os.path.isfile(self.query2rid_file)
            query2rid_data = pd.read_csv(self.query2rid_file, encoding="utf-8", sep=self.data_sep).drop_duplicates()
            self.check_index(query2rid_data)
        elif isinstance(self.query2rid_file, pd.core.frame.DataFrame):
            query2rid_data = self.query2rid_file.copy()
            self.check_index(query2rid_data)
        elif isinstance(self.query2rid_file, dict):
            if self.index_col is None:
                self.query2res = self.query2rid_file
                self.index_data = {k: [v] for k, v in self.query2res.items()}
            else:
                self.query2res, self.index_data = dict(), dict()
                for qas in self.query2rid_file:
                    self.index_data[qas] = []
                    for j in qas:
                        self.query2res.update(j)
                        self.index_data[qas] += list(j.keys())
        else:
            raise ValueError("Error: query2rid_file is invalid")

        # query_col-->res_col
        if query2rid_data is not None:
            # 文本：回答
            self.query2res = query2rid_data.set_index(self.query_col)[self.res_col].to_dict()
            # 索引：文本
            if self.index_col is not None:
                self.index_data = query2rid_data.groupby(self.index_col).apply(
                    lambda x: x[self.query_col].tolist()).to_dict()
            else:
                self.index_data = {k: [v] for k, v in self.query2res.items()}

    def build_index(self, num_tree=10):
        # for self.ix2doc and self.engine
        t0 = time.time()
        logger.info("SE begin create index ...")
        ix = 0
        tmp_list = []
        for doc in self.index_data:
            self.ix2doc[ix] = doc
            if len(doc) > 0:
                tmp_list += [(ix, doc)]
            ix += 1
        if self.tree_rebuild:
            if self.vec_model == "bert":
                vec_list = self.vector_gen.text2vec([d[-1] for d in tmp_list], avg=True, output_array=False)
            elif self.vec_model == "w2v":
                vec_list = [self.vector_gen.text2vec(d[-1].split(), avg=True, output_array=False) for d in tmp_list]
            for j in range(len(tmp_list)):
                self.engine.add_item(tmp_list[j][0], vec_list[j])
            self.engine.build(num_tree)
            logger.info(" Annoy engine with {} items ready".format(self.engine.get_n_items()))
            if isinstance(self.cache_file, str):
                self.engine.save(self.cache_file)
        logger.info("SE finish with time <%.2f>s", time.time() - t0)

    def refresh_index(self):
        if isinstance(self.query2rid_file, str) and os.path.isfile(self.query2rid_file):
            logging.info("Update SE index begin...")
            self.tree_rebuild = True
            self.query2res = {}
            self.index_data = {}
            self.process_data()

            self.ix2doc = {}
            self.engine = AnnoyIndex(self.vec_dim, 'angular')
            self.build_index()
            logging.info("Update SE index finished.")
        else:
            logging.warning("SE data is not static file, index can not update!!！")

    def query_search(self, query_text, res_num=2, doc_format=["list", "text", "split_text"][-1]):
        logger.debug("SE query text: %s", query_text)
        t0 = time.time()
        res = []
        assert isinstance(query_text, (list, tuple, str))
        words = query_text
        query_len = len(words)
        select_num = min(res_num, len(self.ix2doc))
        if query_len > 0:
            res_set = self.engine.get_nns_by_vector(self.vector_gen.text2vec(words, avg=True, output_array=False),
                                                    select_num, search_k=-1, include_distances=True)
            res_cnt = len(res_set)
            if res_cnt < 1:
                raise ValueError("not enough values in result to unpack (expected 2)")
            else:
                if len(res_set[0]) < 1:
                    raise ValueError("no valid value in result")
                res_mapped = [(res_set[0][j], res_set[1][j]) for j in range(len(res_set[0]))]
                for doc_id, score in sorted(res_mapped, key=lambda p: p[-1], reverse=False):
                    r = [self.ix2doc[doc_id]] if self.index_col is None else self.index_data[self.ix2doc[doc_id]]
                    for doc_text in r:
                        if doc_format == "list":
                            out_text = doc_text.split()
                        elif doc_format == "text":
                            out_text = doc_text.replace(" ", "")
                        else:
                            out_text = doc_text
                        # doc_set = set(doc_text.split())
                        # "score": len(doc_set & query_set) / len(doc_set | query_set)
                        res += [{self.res_index_id: doc_id,
                                 self.res_index_col: self.ix2doc[doc_id],
                                 self.res_origin_col: out_text,
                                 self.res_id_col: self.query2res[doc_text],
                                 self.res_sim_col: score}]
        logger.debug("SE search finish with time <%.2f>s", time.time() - t0)
        logger.debug("SE response result: %s", res)
        return res


if __name__ == "__main__":
    from conf.faq_config import PUNCTS_stopwords
    from nlp_utils.nlp_base import ZhNlp
    from text2vector.tfidf import TfIdf

    nlp = ZhNlp(config_lib="jieba", config_dict="/Users/fanzfeng/DocTemp/ZA/w2v/vocab.txt",
                config_stop=PUNCTS_stopwords, seg_out_list=True)
    # idf = TfIdf(corpus_files=["classify_queries"], ngrams=-1, seg_func=nlp.zh_seg, corpus_distinct=True,
    #             base_path=os.path.join(root_path, "/data/service"),
    #             output_model="idf", sparse=False)

    se = SearchEngine(query2rid_file=os.path.join(root_path, "data/service/retrieval.index"),
                      cache_file=os.path.join(root_path, "data/service/w2v_engine.ana"),
                      res_col="qid", query_col="question", index_col="seg",
                      vec_model="w2v", seg_func=nlp.zh_seg, weight_func=None,
                      model_file="/Users/fanzfeng/DocTemp/ZA/w2v/gensim_word2vec2_combine.model")
    # weight_func=idf.text2vec
    se.refresh_index()
    print(se.query_search("我要注销", res_num=10))
    def ss(x):
        r = se.query_search(x, res_num=20)
        return r
    test_df = pd.read_csv(os.path.join(root_path, "data/service/text_label/eva_20191205_.txt"),
                          header=None, names=["text", "qid"], sep="\t")
    test_df["calcu"] = test_df["text"].apply(ss)
    test_df["pred"] = test_df["calcu"].apply(lambda r: [rl["qid"] for rl in r])
    # print("desc score: ", test_df["calcu"].apply(lambda r: r[0]["score"]).describe())
    for n in [1, 3, 5]:
        print("top {} qa_accuracy: ".format(n), test_df.apply(lambda r: r["qid"] in r["pred"][0:n], axis=1).mean())
