# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng
'''
可插拔（链式）文本处理模块
'''

from conf.faq_config import spoken_words, PUNCTS_stopwords, syn_map
from conf.faq_config import conf_path
from nlp_utils.nlp_base import *
from nlp_utils.asr_correct import AsrCorrect


class TextProcessor(object):
    def __init__(self, processors=[]):
        assert len(processors) > 0 and processors[0].input_type == str
        self.processors = [processors[0]]
        for i in range(len(processors)):
            if i >= 1:
                print(processors[i])
                if self.processors[-1].output_type == processors[i].input_type or \
                        self.processors[-1].output_type in processors[i].input_type:
                    self.processors += [processors[i]]

    def process(self, x, out_list=True):
        assert isinstance(x, str)
        p_out = x
        for p in self.processors:
            if len(p_out) <= 0:
                return "" if not out_list else []
            p_out = p.process(p_out)
        if isinstance(p_out, str):
            return p_out
        elif isinstance(p_out, list):
            return p_out if out_list else " ".join(p_out)


if __name__ == "__main__":
    asr_faq_dict = os.path.join(conf_path, "phrase_dict")
    asr_faq_json = os.path.join(conf_path, "phrase_dict.json")
    my_processors = []
    ac = AsrCorrect(dict_path=asr_faq_dict, correct_json=asr_faq_json, acsm_pkl=None)
    my_processors.append(ac)
    my_processors.append(TextClean(drop_list=spoken_words+[" "]))
    my_processors.append(TextReplace(synonym_dict=syn_map))
    my_processors.append(TextSeg(jieba_dict=None, stopwords_config=PUNCTS_stopwords))
    my_processors.append(NeighborUnique(window=2))
    processor = TextProcessor(my_processors)
    print(processor.process("我不是但是，你还是去北京吧"))
