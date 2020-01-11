# -*- coding: utf-8 -*-
# @Author  : fanzfeng

'''
文本分类实现知识库问答系统
1. 此处没有为问题设置答案，只有问题的抽象标签，数据形式<问法，问题标签>，因此返回的时问题标签
2.
'''

from classify.match_classify import *
from text2vector.ngrams import Ngrams

gram_1 = Ngrams(ngrams=1, vocab_file=os.path.join(data_base_path, "qa_dict.char"), input_len=20)


class FaqSearch(object):
    def __init__(self):
        self.input_reply = "输入无效，请重新输入"
        self.greet_reply = "您好，我是小爱，你问我答"
        self.default_reply = "小爱没理解，您可以换个问法试试"
        self.pro_reply = "给您的回答："
        self.list_reply = "您可能问的是："
        self.greet_len = 5
        self.min_sim_score = 0.8
        self.min_range_score = 0.7
        self.max_reply_cnt = 3
        self.greet_keys = ["hi", "hello", "你好", "您好", "在不", "再不", "有人"]
        print("Loading model...")
        self.sim = TextClassifier(seq_len=20, emb_size=64, hidden_size=None, num_class=5, batch_size=16,
                                  epoch_num=5, lr=0.001, keep_rate=0.7, vocab_size=264,
                                  origin_file=os.path.join(data_base_path, "classify_queries"),
                                  model_path=model_save_path, train_test_files="",
                                  model_data_path=os.path.join(data_base_path, "text_label"),
                                  labels_conf=os.path.join(data_base_path, "labels.json"),
                                  model_name="textcnn", vec_models=[gram_1, ])
        print("Finished")

    def rank_res(self, x, top_num=3):
        scores = self.sim.process(x)[-1]
        max_scores = max(scores[0])
        ix_dist = np.argsort(scores)[0]
        v_range = max_scores - scores[0][ix_dist[-2]]
        first_label = self.sim.id2label[str(ix_dist[-1])]
        res = []
        if max_scores >= 0.8 and v_range >= 0.7:
            # if first_label != 'others':
            res = [first_label+"(%.2f)" % max_scores]
        else:
            for j in range(top_num):
                i = -1 * (j + 1)
                if scores[0][ix_dist[i]] >= 0.1:
                    res += [self.sim.id2label[str(ix_dist[i])]+"(%.2f)" % scores[0][ix_dist[i]]]
        return [c.replace('others', '其它') for c in res]

    def bot(self, user_input):
        if not isinstance(user_input, str):
            return self.input_reply
        user_input = user_input.lower().replace(" ", "")
        if len(user_input) < 1:
            return self.input_reply
        if len(user_input) <= self.greet_len:
            for k in self.greet_keys:
                if k in user_input and len(user_input) - len(k) <= 3:
                    return self.greet_reply
        replies = self.rank_res(user_input)
        if len(replies) < 1:
            return self.default_reply
        elif len(replies) == 1:
            return self.pro_reply + " # " + replies[0]
        return " ".join([self.list_reply]+["{}.{}".format(j+1, s) for j, s in enumerate(replies)])


if __name__ == "__main__":
    faq = FaqSearch()
    print(faq.bot("我想知道一下，能告诉我么"))
