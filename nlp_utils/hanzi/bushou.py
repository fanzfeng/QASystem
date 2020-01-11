# -*- coding: utf-8 -*-
# version=3.6.4

import pickle

tool_file = "data.pkl"
girl_keys = ["女", "水", "草"]
girl_base = "美 丽 华 敏 梅 灵 月 琴 佳 玲 慧 静 雅 艳 红 洁 霞 雯 兰 紫 露 翠 韵 香 环 珏 璇 柔 伶 琳 怡 曼 纯".split()
with open(tool_file, 'rb') as fd:
    bs_model = pickle.load(fd)


def query(input_char, default=None):
    # 汉字拆解
    result = bs_model.get(input_char, default)
    if result is None:
        return []
    return result[0]


if __name__ == "__main__":
    print(query(input_char="饭"))
