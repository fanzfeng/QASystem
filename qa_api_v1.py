# -*- coding: utf-8 -*-
# @Author  : fanzfeng

'''
检索-排序实现知识库问答系统
   > 文本处理层：
   > 文本检索层
      简版搜索引擎 qa_search
      语义检索引擎 sem_search
   > 文本排序层
      无监督
      有监督
'''
from conf.faq_config import spoken_words, syn_map, PUNCTS_stopwords
from conf.faq_config import search_type
from nlp_utils.utils import timer, series_unique
from nlp_utils.sim_utils import levenshtein_sim, W2V, TfIdf

from nlp_utils.nlp_base import *
from text2vector.ngrams import Ngrams
from text_process import TextProcessor

from matching.match_new import *


from nlp_utils.sim_utils import logger

char_vocab_file = os.path.join(data_base_path, "qa_dict.char")
word_vocab_file = "/Users/fanzfeng/DocTemp/ZA/w2v/vocab.txt"
w2v_file = "/Users/fanzfeng/DocTemp/ZA/w2v/gensim_word2vec2_combine.model"
engine_file = os.path.join(data_base_path, "w2v_engine.ana")
origin_index_file = os.path.join(data_base_path, "retrieval.index")


class FaqSearch(object):
    @timer(prefix='FaqSearch')
    def __init__(self, rank_model="edit_distance"):
        self.my_processors = []
        self.my_processors.append(TextClean(drop_list=spoken_words + [" "]))
        self.my_processors.append(TextReplace(synonym_dict=syn_map))
        self.my_processors.append(TextSeg(jieba_dict=word_vocab_file, stopwords_config=PUNCTS_stopwords,
                                          seq_labels=None, label_mark=None))
        self.processor = TextProcessor(self.my_processors)

        logger.info("SE load index data and search model ...")
        if search_type == "semantic":
            from retrieval.sem_search import SearchEngine
            se_nlp = ZhNlp(config_lib="jieba", config_dict=word_vocab_file, config_stop=PUNCTS_stopwords,
                           seg_out_list=True)
            self.engine = SearchEngine(query2rid_file=origin_index_file, cache_file=engine_file,
                                       res_col="qid", query_col="question", index_col="seg", vec_model="w2v",
                                       model_file=w2v_file, seg_func=se_nlp.zh_seg, weight_func=None)
        else:
            from retrieval.qa_search import SearchEngine
            self.engine = SearchEngine(query2rid_file=origin_index_file, res_col="qid", query_col="question",
                                       index_col="seg")
        logger.info("SE index and rank model is ready")

        self.score_col = "rank_score"
        self.res_qcol = self.engine.res_origin_col
        if rank_model == "edit_distance":
            self.rank_func = levenshtein_sim
        elif rank_model == "simnet":
            seg_gram_1 = Ngrams(ngrams=1, vocab_file=char_vocab_file, input_len=20)
            sim_model = SimMatch(seq_len=20, net_params={"hidden_size": 512, "vocab_size": 1605, "emb_size": 256},
                                 vec_models=[seg_gram_1], match_model="bow",
                                 batch_size=None, epoch_num=None, lr=None, keep_rate=None, files_split=None,
                                 engine=self.engine, model_path=model_save_path)
            self.rank_func = sim_model.process

        self.default_reply = "小爱没理解，您可以换个问法试试"
        self.pro_reply = "给您的回答："
        self.list_reply = "您可能问的是："
        self.greet_reply = "您好，我是小爱，你问我答"
        self.greet_len = 5
        self.min_sim_score = 0.2
        self.min_range_score = 0.08
        self.max_reply_cnt = 3
        self.greet_keys = ["hi", "hello", ]#"你好", "您好", "在不", "再不", "有人"

        self.check_params()

    def check_params(self):
        pass

    # @timer(prefix="api")
    def rank_res(self, query_text, recall_num=50, min_score=0.3, mark=False):
        if len(query_text) < 1:
            return []
        text = self.processor.process(query_text)
        if len(text) < 1:
            # logger.info("Processor invalid string <{}>".format(query_text))
            return []
        text = "".join(text)
        recall_res = self.engine.query_search(text, res_num=recall_num)
        if len(recall_res) < 1:
            return []
        if self.rank_func is None:
            return recall_res
        for j in range(len(recall_res)):
            recall_res[j][self.score_col] = self.rank_func(text, recall_res[j][self.res_qcol])
        ranked_res = sorted(recall_res, key=lambda r: r[self.score_col], reverse=True)
        # debug method
        if mark:
            return ranked_res
        # formal method
        if ranked_res[0][self.score_col] < min_score:
            return []
        return ranked_res[0:1]

    @timer(prefix="One Request")
    def bot(self, user_input):
        if not isinstance(user_input, str):
            return self.default_reply
        user_input = user_input.lower().replace(" ", "")
        if len(user_input) < 1:
            return self.default_reply
        if len(user_input) <= self.greet_len:
            for k in self.greet_keys:
                if k in user_input:
                    return self.greet_reply
        replies = self.rank_res(user_input, recall_num=10, mark=True)
        if len(replies) < 1:
            return self.default_reply
        elif replies[0]['rank_score'] < self.min_sim_score:
            return self.default_reply
        elif replies[0]['rank_score'] - replies[1]['rank_score'] < self.min_range_score:
            replies_cnt = min(len(replies), self.max_reply_cnt)
            replies_orgin = series_unique([s["qid"] for s in replies[0:replies_cnt]])
            return " ".join([self.list_reply]+["{}.{}".format(j+1, s) for j, s in enumerate(replies_orgin)])
        else:
            return self.pro_reply+" # "+replies[0]["qid"]


if __name__ == "__main__":
    faq = FaqSearch(rank_model="simnet")
    print(faq.bot("有人在不"))
