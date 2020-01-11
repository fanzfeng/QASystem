# -*- coding: utf-8 -*-
'''
共用的参数配置模块
'''
import os

search_type = ["semantic", "inverted"][0]

# data_base_path = "data/data_faq"
pro_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
conf_path = os.path.join(pro_path, "conf/files")

file_spoken_words = os.path.join(conf_path, "slu_words_del")# 口语冗余词
file_stop_dict = os.path.join(conf_path, "stop_words.txt")# 停用词

spoken_words = [w.strip() for w in open(file_spoken_words, "r", encoding="utf-8").readlines()]
# print("Spoken words: ", spoken_words[0:3])

# 定制场景相关的近义词, 一般是产品相关
syn_dict = {"会员": ["线上会员", "内部会员"]}
syn_map = {}
try:
    for v, us in syn_dict.items():
        for u in us:
            syn_map[u] = v
except Exception as e:
    print("synonym exists duplicate keys: ", e)

PUNCTS_stopwords = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' + '！“”￥…‘’（），–—－。、：；《》？【】·√◆⾃●'
