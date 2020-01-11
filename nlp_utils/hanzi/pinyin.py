# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng
'''
拼音错误来源：
1. 语言习惯
2. 键盘布局（26按键/9宫格）

纠错方法：
1. 转拼音，求编辑距离
'''

from xpinyin import Pinyin
from nlp_utils.sim_utils import levenshtein_sim

# 常见拼音错误（语言习惯）
pinyin_dict = {"0": [["zh", "z"], ["ch", "c"], ["sh", "s"], ["n", "l"], ["h", "f"], ["r", "l"], ["k", "g"]],
               "1": [["uang", "uan"], ["iang", "ian"], ["ang", "an"], ["eng", "en"], ["ing", "in"]]
               }
p = Pinyin()


def pinyin_correct(py):
    if len(py) <= 0:
        return "string is null"
    for k in pinyin_dict[0]:
        n = len(k[0])
        if len(py) > n and py[0:n] == k[0]:
            return k[1]+py[n:]
    for k in pinyin_dict[1]:
        n = len(k[0])
        if len(py) > n and py[-n:] == k[0]:
            return py[:-n]+k[1]
    return py


if __name__ == "__main__":
    print(p.get_pinyin("上海", "009"))
    print(p.get_pinyin(u"上海", tone_marks='marks'), "< >", p.get_pinyin(u"上海", tone_marks='numbers'))
    print(pinyin_correct('chuang'))
