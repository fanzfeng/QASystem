# -*- coding: utf-8 -*-
# @Date  : 2019/1/2
# @Author  : fanzfeng
'''
简版搜索引擎/词典检索
    input:
        1. query2rid_file接受已经完成预处理和切好词【空格分隔】的(query, rid)pair对，rid表示的是问题表示，对应唯一的answer
        2. query2rid_file可以为文件名，也可以为dataframe
    control of params:
        1. ix2doc为doc的索引，key_dict为word:list of doc_ix倒排索引
        2. 一个index对应多个query、多个answer，query与answer一一对应
        3. index_col为待检索文本列名, query_col为返回的文本列名
'''

import os
import time
import logging
from collections import Counter
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)


class SearchEngine(object):
    def __init__(self, query2rid_file, res_col="answer", query_col="question", index_col=None):
        self.index_col = index_col
        self.query2rid_file = query2rid_file
        self.res_col = res_col
        self.query_col = query_col

        self.res_index_col = "index_text"
        self.res_origin_col = "query_text"
        self.res_sim_col = "score"
        self.res_id_col = "qid"
        self.res_index_id = "query_ix"

        self.query2res = {} # query_text2std_id(answer_id)
        self.index_data = {} # query_text2id_list
        self.process_data()

        self.ix2doc = {}  # qid2query_text
        self.key_dict = {} # keyword2qid_list
        self.build_index()

    def process_data(self):
        query2rid_data = None
        if isinstance(self.query2rid_file, str) and os.path.isfile(self.query2rid_file):
            query2rid_data = pd.read_csv(self.query2rid_file, encoding="utf-8", sep=",").drop_duplicates()
        elif isinstance(self.query2rid_file, pd.core.frame.DataFrame):
            query2rid_data = self.query2rid_file.copy()
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

        # query_col-->res_col
        if query2rid_data is not None:
            self.query2res = query2rid_data.set_index(self.query_col)[self.res_col].to_dict()
            if self.index_col is not None:
                self.index_data = query2rid_data.groupby(self.index_col).apply(
                    lambda x: x[self.query_col].tolist()).to_dict()
            else:
                self.index_data = {k: [v] for k, v in self.query2res.items()}

    def build_index(self):
        t0 = time.time()
        logger.info("SE begin create index ...")
        ix = 0
        for doc in self.index_data:
            self.ix2doc[ix] = doc
            words = doc.split()
            for k in words:
                if k not in self.key_dict:
                    self.key_dict[k] = [ix]
                elif ix not in self.key_dict[k]:
                    self.key_dict[k] += [ix]
            ix += 1
        logger.info("SE finish with time %s", time.time() - t0)

    def refresh_index(self):
        if isinstance(self.query2rid_file, str) and os.path.isfile(self.query2rid_file):
            self.index_data = {}
            self.query2res = {}
            self.process_data()

            self.ix2doc = {}
            self.key_dict = {}
            self.build_index()
            logging.info("SE update index finished")
        else:
            logging.warning("SE data is not static file, index can not update!!！")

    def query_search(self, query_text, res_num=2, doc_format=["list", "text", "split_text"][-1]):
        logger.debug("SE query text: %s", query_text)
        t0 = time.time()
        res = []
        assert isinstance(query_text, (list, tuple, str))
        if isinstance(query_text, str):
            words = query_text.split()
        else:
            words = query_text
        query_len = len(words)
        query_set = set(words)
        if query_len > 0:
            doc_related = []
            # weights = [1/query_len]*query_len
            for w in query_set:
                doc_related += self.key_dict.get(w, [])
            res_len = len(doc_related)
            if res_len > 0:
                doc_freq = Counter(doc_related)
                doc_res = doc_freq.most_common()[0:min(res_num, res_len)]
                for doc_id, freq in doc_res:
                    # index_col is none query-res, not index-res
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
                                 self.res_id_col: self.query2res[doc_text]}]
        logger.debug("SE search finish with time %s", time.time() - t0)
        logger.debug("SE response result: %s", res)
        return res
