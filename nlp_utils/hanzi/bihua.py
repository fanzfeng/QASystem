# -*- coding: utf-8 -*-
# version=3.6.4
'''
资源参考：
    Unicode汉字笔画顺序表: https://download.csdn.net/download/bao110908/363125
实现参考(基于汉字的字形编码结果计算计算相似)：
   https://github.com/i4tv/i4tv.git
举例如下：
七      hz      6313
万      hzp     25941
丈      hpn     305
三      hhh     26761
上      shh     284918
下      hsn     163638
不      hpsn    3
与      hzh     202250

'''
# data_lines = open(u'汉字编码表 gbk unicode.txt').readlines()
#
# query_dict = {}
#
# for line in data_lines[7:]:
#     l = line.strip().split()
#     unicode_mark = unichr(int(l[4], 16))
#     bihua = l[6]
#     query_dict[unicode_mark] = bihua
#
# print u'收录汉字个数:', len(data_lines)
#
# for s in u'你好世界！':
#     print s, query_dict.get(s, -1)
