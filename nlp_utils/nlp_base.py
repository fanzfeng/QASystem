# -*- coding: utf-8 -*-
# @Date  : 2019/4/25
# @Author  : fanzfeng

import os
import re
import jieba
import jieba.posseg as pseg
import platform


class TextClean(object):
    def __init__(self, drop_list):
        self.input_type = str
        self.output_type = str
        self.drop_list = sorted(drop_list, key=len, reverse=True)

    def process(self, str_origin):
        if isinstance(str_origin, self.input_type):
            str_tmp = str_origin
            for w in self.drop_list:
                str_tmp = str_tmp.replace(w, "")
            return str_tmp


class TextReplace(object):
    def __init__(self, synonym_dict):
        self.input_type = str
        self.output_type = str
        if isinstance(synonym_dict, dict) and len(synonym_dict) > 0:
            self.syn_map = {k: [v] if isinstance(v, str) else v for k, v in synonym_dict.items()}
            self.syn_map = {k: sorted(v, key=len, reverse=True) for k, v in self.syn_map.items()}

    def process(self, str_origin):
        if self.syn_map and isinstance(str_origin, self.input_type):
            for k, v in self.syn_map.items():
                for w in v:
                    str_origin = str_origin.replace(w, k)
            return str_origin


class NeighborUnique(object):
    def __init__(self, window=2):
        self.input_type = list
        self.output_type = list
        self.window = window
        self.words_ignore = ["D"]

    def process(self, words):
        if isinstance(words, self.input_type):
            new_words = []
            for j in range(len(words)):
                w = words[j]
                # if w not in words[max(0, j - dup_window):j]:
                if w in self.words_ignore or sum(w in w_bf for w_bf in words[max(0, j - self.window):j]) <= 0:
                    new_words.append(w)
            return new_words


class TextSeg(object):
    def __init__(self, jieba_dict, stopwords_config, seq_labels=["ns"], label_mark="D"):
        self.input_type = str
        self.output_type = list
        self.nlp = ZhNlp(config_lib="jieba", config_dict=jieba_dict, config_stop=stopwords_config, seg_out_list=True)
        self.ner_format = False
        if seq_labels and label_mark:
            self.ner_format = True
            self.seq_labels = seq_labels
            self.label_mark = label_mark

    def process(self, x):
        if isinstance(x, self.input_type) and len(x) > 0:
            if not self.ner_format:
                return self.nlp.zh_seg(x)
            res_ner = self.nlp.zh_ner(x)[0]
            new_words = []
            for i in range(len(res_ner[0])):
                if res_ner[-1][i] in self.seq_labels:
                    new_words.append(self.label_mark)
                else:
                    new_words.append(res_ner[0][i])
            return new_words


class NgramsSeg(object):
    def __init__(self, ngrams=1, out_list=True):
        self.input_type = [list, str]
        self.output_type = list if out_list else str
        self.k = ngrams
        self.out_list = out_list

    def text_seg(self, x):
        x_list = [""]
        if len(x) > 0:
            x = [str(s) for s in x]
            x_len = len(x)
            if x_len >= self.k:
                x_list = []
                for i in range(x_len-self.k+1):
                    x_list += ["".join(x[i:i+self.k])]
        return x_list if self.out_list else " ".join(x_list)

    def process(self, x):
        self.text_seg(x)


class ZhNlp(object):
    def __init__(self, config_lib="ltp", config_dict=None, config_stop=None, config_dir=None, seg_out_list=False):
        self.input_type = str
        self.config_dir = config_dir
        if config_dir is None:
            self.config_dir = 'E:/Data/' if 'windows' in platform.architecture()[1].lower() else '/users/fanzfeng/Data/'

        self.stop_config = False
        if config_stop is not None and isinstance(config_stop, str) and os.path.exists(config_stop):
            self.stop_config = True
            with open(config_stop, "r", encoding="utf-8") as fp:
                self.stop_words = [k.strip() for k in fp.readlines() if len(k.strip()) > 0]
        elif isinstance(config_stop, (list, tuple, set)) and len(config_stop) > 0:
            self.stop_config = True
            self.stop_words = config_stop

        self.all_cut = False
        self.seg_out_list = seg_out_list

        self.config_lib = config_lib
        if config_lib == "jieba":
            self.jieba_ner = "nr ns nt m".split()
            if config_dict is not None and isinstance(config_dict, str) and os.path.exists(config_dict):
                jieba.load_userdict(config_dict)
            self.seg = jieba.cut
            self.pos_seg = pseg.cut
        elif config_lib == "ltp":
            import pyltp
            self.segmentor = pyltp.Segmentor()
            if config_dict is not None and isinstance(config_dict, str) and os.path.exists(config_dict):
                self.segmentor.load_with_lexicon(os.path.join(self.config_dir, "ltp_data_v3.4.0/cws.model"), config_dict)
            else:
                self.segmentor.load(os.path.join(self.config_dir, "ltp_data_v3.4.0/cws.model"))
            self.seg = self.segmentor.segment
            self.postagger = pyltp.Postagger()
            self.text_splitter = pyltp.SentenceSplitter.split
            self.postagger.load(os.path.join(self.config_dir, "ltp_data_v3.4.0/pos.model"))
            self.recognizer = pyltp.NamedEntityRecognizer()
            self.recognizer.load(self.config_dir + "ltp_data_v3.4.0/ner.model")

    def split_sentence(self, doc, delimiters=list("。？！")):
        if self.config_lib == "ltp":
            sents = self.text_splitter(doc)
            return list(sents)
        else:
            return re.split("|".join(delimiters), doc)

    def ltp_close(self):
        if self.config_lib == "ltp":
            self.segmentor.release()
            self.postagger.release()
            self.recognizer.release()

    def zh_seg(self, text_input, drop_stop=True, all_cut=False, output_postags=False, out_list=False):
        if isinstance(text_input, str):
            text_seq = [text_input]
        elif isinstance(text_input, (list, tuple)):
            text_seq = text_input
        if text_seq:
            grams_series = []
            for x in text_seq:
                series_words = (self.seg(x, cut_all=self.all_cut) if self.config_lib == "jieba" else self.seg(x))
                if self.stop_config:
                    grams_series += [[w for w in series_words if w not in self.stop_words]]
                else:
                    grams_series += [list(series_words)]
            if not self.seg_out_list:
                grams_series = [" ".join(s) for s in grams_series]
            return grams_series[0] if isinstance(text_input, str) else grams_series

    def zh_pos(self, text_input):
        if isinstance(text_input, str):
            text_seq = [text_input]
        elif isinstance(text_input, (list, tuple)):
            text_seq = text_input
        if text_seq:
            grams_series = []
            for s in text_seq:
                if self.config_lib == "ltp":
                    word_list = list(self.seg(s))
                    postags = self.postagger.postag(word_list)
                    ptag_list = list(postags)
                    if len(word_list) == len(ptag_list):
                        grams_series += [(word_list, ptag_list)]
                else:
                    seg_res = [(w, p) for w, p in self.pos_seg(s)]
                    grams_series += [([k[0] for k in seg_res], [k[1] for k in seg_res])]
            out_put = []
            if self.stop_config:
                for wlist, plist in grams_series:
                    w_list, p_list = [], []
                    for i in range(len(wlist)):
                        if wlist[i] not in self.stop_words:
                            w_list += [wlist[i]]
                            p_list += [plist[i]]
                    out_put += [(w_list, p_list)]
            else:
                for wlist, plist in grams_series:
                    w_list, p_list = [], []
                    for i in range(len(wlist)):
                        w_list += [wlist[i]]
                        p_list += [plist[i]]
                    out_put += [(w_list, p_list)]
            return out_put[0] if isinstance(text_input, str) else out_put

    def zh_ner(self, text):
        if isinstance(text, str) and len(text) > 0:
            if self.config_lib == "ltp":
                seg_res, pos_res = self.zh_pos(text)
                netags = self.recognizer.recognize(seg_res, pos_res)
                ner_res = (seg_res, netags)
                return ner_res
            else:
                return self.zh_pos(text)


if __name__ == "__main__":
    pass
