# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng

import json
import pickle
import time
from functools import wraps


def series_unique(series):
    new_series = []
    for s in series:
        if s not in new_series:
            new_series += [s]
    return new_series


def save_json(data_series, save_file, complex_series=False):
    with open(save_file, "w", encoding="utf-8") as fp:
        if not complex_series:
            json.dump(data_series, fp, indent=4, ensure_ascii=False)
        else:
            for d in data_series:
                json.dump(d, fp, ensure_ascii=False)
                fp.write("\n")


def save_pkl(data, save_file):
    with open(save_file, "wr") as fp:
        pickle.dump(data, fp)
        print("Success save data to {}".format(save_file))


def load_pkl(save_file):
    with open(save_file, "rb") as fp:
        return pickle.load(fp)


def timer(prefix='', logger=None, level=None):
    def decorator(func):
        @wraps(func)
        def f(*args, **kwargs):
            before = time.time()
            rv = func(*args, **kwargs)
            after = time.time()
            cost = after - before
            cost_str = '<%.2f> min' % (cost/60) if cost > 60 else '<%.2f> s' % cost if cost >= 1.0 else '<%.2f> ms' % (cost * 1000)
            if logger is not None and level is not None:
                logger.log(level, 'finished: (%s %s), cost: %s' % (prefix, func.__name__, cost_str))
            else:
                print('%s %s finished, cost time: %s' % (prefix, func.__name__, cost_str))
            return rv
        return f
    return decorator
