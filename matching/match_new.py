# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng

'''
(！！！置顶说明：检索对词向量和对应的分词词表有依赖，可选择下载开源的或者利用gensim训练一个自己语料下的词向量)

20191123更新：
1）文本向量表示器vec_models可插拔，维度同占位tensor、batch分割保持一致
  - 场景：多个特征多个网络后concat工作预测输出（网络融合方法）
2）eval_metrics可插拔，测试集效果直观可见
3) batch generator修改为字典形式，支持新增
4）去除特征数据存储，读数据-->分数据-->特征化, 一步到位

语义匹配 semantic match:
1) 索引数据--->语义网络训练及测试数据(索引数据位以下简称匹配数据)
2) 匹配数据的读取--->batch
3) train根据静态的匹配数据(文件格式)训练匹配网络, 并保存模型
4) process加载静态模型, 实时处理文本pair对(输入: text pair, 输出: sim score)
5) refresh刷新训练数据和模型
'''

import pandas as pd
import sys, os, time
from collections import OrderedDict

root_path = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(root_path)
sys.path.append(root_path)
from retrieval.sem_search import SearchEngine
from nets.network import nets_dict, tf, np
import metrics

data_base_path = os.path.join(root_path, "data/service")
data_model_path = os.path.join(data_base_path, "text_pairs")

model_dir = os.path.join(root_path, "models")
model_save_path = model_dir+"/{m}_match"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


class SimMatch(object):
    def __init__(self, model_path, batch_size, epoch_num, lr, keep_rate, seq_len=None, net_params=None,
                 origin_file=None, files_split=None, match_model="bert", vec_models=None, engine=None, recall_num=5,
                 eval_metrics=None):
        '''
        :param model_path: 模型存放路径
        :param origin_file: 原始文本-类别文件/问法-问题
        :param files_split: 处理好的训练集和测试集
        :param match_model: 排序/匹配模型
        :param vec_models: 特征提取器（list，可插拔）
        :param engine: 检索引擎
        :param eval_metrics: 评价指标（list，可插拔）
        '''
        self.r_state = 666
        assert isinstance(vec_models, list) and len(vec_models) > 0
        if match_model not in nets_dict:
            raise ValueError("valid model must in: {}".format(" ".join([k for k in nets_dict.keys()])))
        self.model_name = match_model
        self.model_path = model_path.format(m=match_model)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.max_models_num = 7
        self.net_params = net_params
        self.ignore_std_queries = ["others"]

        self.featurizers = []
        for m in vec_models:
            if isinstance(m, list):
                self.featurizers += m
            else:
                self.featurizers.append(m)
        self.engine = engine

        if eval_metrics:
            assert isinstance(eval_metrics, list) and len(eval_metrics) > 0
            self.eval_metrics = OrderedDict()
            for mobj in eval_metrics:
                mobj = mobj.lower()
                if '@' in mobj:
                    mt_key, mt_val = mobj.split('@', 1)
                    self.eval_metrics[mobj] = metrics.get(mt_key)(int(mt_val))
                else:
                    self.eval_metrics[mobj] = metrics.get(mobj)

        self.origin_file, self.files_split = origin_file, files_split
        self.sep = "\t"
        self.query_col, self.label_col = "question", "qid"
        self.train_rate = 0.7 # 训练测试数据分布
        self.key_cols = [self.label_col, self.query_col]

        self.per_docs_num = recall_num
        self.input_cols = ["text_{}".format(i) for i in range(2)]
        self.num_feed_x = len(self.featurizers)*len(self.input_cols)
        self.model_col = "label"
        self.sample_dist = 1.0 # 训练集标签分布
        self.cols = self.input_cols+[self.model_col]

        self.query2qid, self.train_num, self.eva_num = None, None, None

        self.seq_len = seq_len
        self.num_class = 2
        self.keep_rate = keep_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr
        self.tf_dtypes, self.np_dtypes = [], []
        for m in self.featurizers:
            self.tf_dtypes += [tf.float32 if len(m.output_shape) > 2 else tf.int32]*len(self.input_cols)
            self.np_dtypes += [np.float32 if len(m.output_shape) > 2 else np.int32]*len(self.input_cols)
        self.tf_dtypes.append(tf.int32)
        self.np_dtypes.append(np.int32)
        self.input_tensors = []
        self.test_y, self.keep_prob = None, None
        self.define_tensor()
        self.net_loss, self.one_hot_labels, self.pred_prob = None, None, None
        self.net_init()

        self.session = None
        model_save = tf.train.get_checkpoint_state(self.model_path)
        if model_save and model_save.model_checkpoint_path:
            print("Loading matching model...")
            # tf.reset_default_graph()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            try:
                saver.restore(self.session, model_save.model_checkpoint_path)
                print("Rank model is ready")
            except:
                print("Load rank model Failed !!")
        else:
            print("Rank model not exists")

    def define_tensor(self):
        k = 0
        for i in range(len(self.featurizers)):
            for j in range(len(self.input_cols)):
                self.input_tensors.append(tf.placeholder(self.tf_dtypes[k], self.featurizers[i].output_shape,
                                                         name="input_{}_{}".format(i, j)))
            k += 1
        self.test_y = tf.placeholder(self.tf_dtypes[-1], [None, self.num_class], name="input_label")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def net_init(self):
        tensor_dict = {"input": self.input_tensors, "output": self.test_y, "keep_rate": self.keep_prob}
        net_conf = {"n_class": self.num_class, "seq_len": self.seq_len}
        if isinstance(self.net_params, dict) and len(self.net_params) > 0:
            net_conf.update(self.net_params)
        net_sim = nets_dict[self.model_name](conf=net_conf)
        if net_sim:
            self.pred_prob, self.one_hot_labels, self.net_loss = net_sim.net(tensor_dict)

    @staticmethod
    def under_sample(data, label_col, sample_dist, r_state):
        if label_col in data.columns and data[label_col].value_counts().shape[0] == 2:
            ix_p, ix_n = list(set(data[label_col]))
            pos_num, neg_num = sum(data[label_col] == ix_p), sum(data[label_col] == ix_n)
            if neg_num > pos_num:
                sample_num = int(min(neg_num, sample_dist*pos_num))
                neg_df = data[data[label_col] == ix_n].sample(n=sample_num, random_state=r_state)
                pos_df = data[data[label_col] == ix_p]
            else:
                sample_num = int(min(pos_num, sample_dist * neg_num))
                pos_df = data[data[label_col] == ix_p].sample(n=sample_num, random_state=r_state)
                neg_df = data[data[label_col] == ix_n]
            new_df = pd.concat([neg_df, pos_df], ignore_index=True)
            return new_df

    def gen_data(self, origin_file):
        print("\nsplit index data, get train and eva queries data")
        db = pd.read_csv(origin_file, sep=self.sep)
        origin_num, cols_cnt = len(db), db.shape[1]
        key_cnt = len(self.key_cols)
        assert cols_cnt >= key_cnt
        db.columns = self.key_cols if cols_cnt == key_cnt else self.key_cols+[str(c) for c in range(cols_cnt-key_cnt)]

        df_queries = db[self.key_cols].drop_duplicates()
        unique_num = len(df_queries)
        train_df = df_queries.sample(frac=self.train_rate, random_state=self.r_state)
        eva_df = df_queries[~df_queries.index.isin(train_df.index)]
        print("  origin num {}, unique {}, split {} & {}".format(origin_num, unique_num, len(train_df), len(eva_df)))
        return {"train": train_df, "eva": eva_df}

    def load_data(self, eva_sample=False):
        print("\nget train and eva queries data")
        if isinstance(self.files_split, dict) and len(self.files_split) > 0:
            queries_dict = {}
            for k, v in self.files_split.items():
                if os.path.exists(v):
                    if "eva" in k:
                        queries_dict[k] = pd.read_csv(v, sep=self.sep, header=None, names=[self.query_col,
                                                                                           self.label_col])
                    else:
                        queries_dict[k] = pd.read_csv(v, sep=self.sep).drop_duplicates()
                    if eva_sample and len(queries_dict[k]) > 6000:
                        queries_dict[k] = queries_dict[k].sample(n=6000, random_state=self.r_state)
        if queries_dict:
            return queries_dict

    def featurizer(self, t2v, text_list):
        if t2v.name == "dict":
            return np.array([t2v.text2vec(s) for s in text_list])
        if t2v.name == "w2v":
            assert self.seq_len is not None
            return np.array([t2v.text2vec(s, avg=False, output_array=False) for s in text_list])
        elif t2v.name == "bert":
            num = len(text_list)
            if num > 100:
                t0 = time.time()
            vec_out = np.array(t2v.text2vec(text_list, avg=False, output_array=False))
            if num > 100:
                print("Text samples num {}, use time {}s".format(num, int(time.time() - t0)))
            return vec_out

    def get_docs(self, df, df_file, cache=False):
        if cache and os.path.exists(df_file):
            full_df = pd.read_pickle(df_file)
            print("Load feature data {}".format(df_file))
        else:
            data_list = []
            q_id, cj = 0, 0
            for j in range(df.shape[0]):
                l_query, l_y = df[self.query_col].iloc[j], df[self.label_col].iloc[j]
                if l_y not in self.ignore_std_queries:
                    q_id += 1
                    res_tmp = self.engine.query_search(l_query, res_num=self.per_docs_num)
                    for r in res_tmp:
                        if l_query != r[self.engine.res_origin_col]:
                        # if True:
                            cj += 1
                            data_list += [[q_id, l_query, r[self.engine.res_origin_col],
                                           1*(l_y == r[self.engine.res_id_col]), cj]]
            if "train" in df_file:
                new_df = pd.DataFrame(data_list, columns=["id"]+self.cols+["id_"])
                full_df = self.under_sample(new_df, sample_dist=self.sample_dist, r_state=self.r_state,
                                            label_col=self.model_col)
            else:
                full_df = pd.DataFrame(data_list, columns=["id"]+self.cols+["id_"])
            full_df["y_vec"] = full_df[self.model_col].apply(lambda y: [1, 0] if y <= 0 else [0, 1])
            for i in range(len(self.featurizers)):
                if self.featurizers[i].name == "bert" and self.model_name == "bert":
                    full_df["full_text"] = full_df.apply(
                        lambda sr: "[sep]".join([str(s) for s in sr[self.input_cols].tolist()]), axis=1)
                    full_df["vec"] = self.featurizer(self.featurizers[i], full_df["full_text"].tolist()).tolist()
                for j in range(len(self.input_cols)):
                    c = self.input_cols[j]
                    if self.featurizers[i].name == "bert" and self.model_name == "bert":
                        full_df["vec_{}_{}".format(i, j)] = full_df["vec"]
                    else:
                        full_df["vec_{}_{}".format(i, j)] = self.featurizer(self.featurizers[i], full_df[c].tolist()).tolist()
            full_df.to_pickle(df_file)
        return [full_df, [0]+full_df.groupby("id").apply(lambda ss: ss["id_"].max()).tolist()]

    def batch_iter(self, data, eva=False):
        if eva:
            df, cnts_list = data
            x_vec_list = []
            for i in range(len(self.featurizers)):
                for j in range(len(self.input_cols)):
                    x_vec_list += [np.array(df["vec_{}_{}".format(i, j)].tolist())]
            yield x_vec_list + [np.array(df["y_vec"].tolist()), cnts_list]
        else:
            df = data[0].sample(frac=1.0, random_state=self.r_state)
            obs = len(df)
            batch_num = int(obs/self.batch_size)
            data_feed = []
            for i in range(len(self.featurizers)):
                for j in range(len(self.input_cols)):
                    data_feed += [np.asarray(df["vec_{}_{}".format(i, j)].tolist())]
            data_feed.append(np.array(df["y_vec"].tolist()))
            for j in range(batch_num):
                yield [v[j*self.batch_size:min(obs, j*self.batch_size+self.batch_size)] for v in data_feed]

    def rank_metric(self, y_real, y_model, list_cnt):
        res = {k: 0 for k in self.eval_metrics}
        queries_cnt = len(list_cnt) - 1
        for k, eval_func in self.eval_metrics.items():
            for lc_idx in range(queries_cnt):
                pre = list_cnt[lc_idx]
                suf = list_cnt[lc_idx + 1]
                res[k] += eval_func(y_true=y_real[pre:suf], y_pred=y_model[pre:suf])
        print('  rank evaluation\t%s' % ('\t'.join(['%s=%f' % (k, v/queries_cnt) for k, v in res.items()])))
        return res

    def train(self, cache=True):
        print("\nget data and caculate feature...")
        data = self.load_data(eva_sample=False)
        if data:
            print("  queries data: {}".format('\t'.join(['%s=%d' % (k, len(v)) for k, v in data.items()])))
            data_full = {k: self.get_docs(v, os.path.join(data_model_path, k+".pkl"), cache=cache) for k, v in data.items()}
            print("  flatten queries data: {}".format("\t".join(['%s=%d' % (k, len(v[0])) for k, v in data_full.items()])))
            print("  train label dist:\n", data_full["train"][0][self.model_col].value_counts())

            print("data for semantic network ready")
            pred_index = tf.argmax(self.pred_prob, 1)
            pred_score = self.pred_prob[:, 1]
            true_index = tf.argmax(self.one_hot_labels, 1)
            correct_pred = tf.equal(pred_index, true_index)
            acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
            if "bcnn" in self.model_name:
                optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.net_loss)
            else:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.net_loss)

            print("Begin Training...")
            saver = tf.train.Saver(max_to_keep=self.max_models_num)
            self.session = tf.Session()
            j = 0
            self.session.run(tf.global_variables_initializer())
            for e in range(self.epoch_num):
                for input_all in self.batch_iter(data_full["train"], eva=False):
                    feeds = {self.keep_prob: self.keep_rate, self.test_y: input_all[-1].astype(self.np_dtypes[-1])}
                    for k in range(self.num_feed_x):
                        feeds[self.input_tensors[k]] = input_all[k].astype(self.np_dtypes[k])
                    cost, _, train_acc = self.session.run([self.net_loss, optimizer, acc], feed_dict=feeds)
                    j += 1
                    if j % 10 == 0:
                        print(" " * 4, ">>> train loss: {} accuracy: {}".format(cost, train_acc))
                for eva_input in self.batch_iter(data_full["eva"], eva=True):
                    eva_feeds = {self.keep_prob: 1.0, self.test_y: eva_input[-2].astype(self.np_dtypes[-1])}
                    for k in range(self.num_feed_x):
                        eva_feeds[self.input_tensors[k]] = eva_input[k].astype(self.np_dtypes[k])
                    # res_eva = self.session.run([acc, true_index, pred_index], feed_dict=eva_feeds)
                    res_eva = self.session.run([acc, true_index, pred_score], feed_dict=eva_feeds)
                    print("epoch %d loss on validation data: %f | accuracy --> : %f" % (e+1, cost, res_eva[0]))
                    self.rank_metric(y_real=res_eva[1], y_model=res_eva[2], list_cnt=eva_input[-1])
                saver.save(self.session, self.model_path + "/model_e{}.ckpt".format(e+1))
            print("Train Finished")

    def process(self, text1, text2):
        text_inputs = [text1, text2]
        if self.session is not None:
            p_inputs = []
            for i in range(len(self.featurizers)):
                shapes = self.featurizers[i].output_shape
                shapes = [1]+shapes[1:]
                if self.featurizers[i].name == "bert" and self.model_name == "bert":
                    vec = np.array(self.featurizers[i].text2vec("{}[sep]{}".format(text1, text2)))
                for j in range(len(self.input_cols)):
                    if self.featurizers[i].name == "bert" and self.model_name == "bert":
                        p_inputs.append(vec.reshape(shapes))
                    else:
                        p_inputs.append(self.featurizer(self.featurizers[i], [text_inputs[j]]).reshape(shapes))
            p_feeds = {self.keep_prob: 1.0,
                       self.test_y: np.array([0, 1]).reshape(1, self.num_class).astype(self.np_dtypes[-1])}
            for k in range(self.num_feed_x):
                p_feeds[self.input_tensors[k]] = p_inputs[k].astype(self.np_dtypes[k])
            pred_ = self.session.run([self.pred_prob], feed_dict=p_feeds)
            return pred_[0][0][-1]

    def refresh(self, cache=True):
        print("Refreshing data and model...")
        self.train(cache=cache)
        print("Refresh Finished")


if __name__ == "__main__":
    # from text2vector.vector_model import Text2Vector
    # bert_nlp = Text2Vector(w2v_file="bert_api")
    from conf.faq_config import PUNCTS_stopwords
    from nlp_utils.nlp_base import ZhNlp
    from text2vector.tfidf import TfIdf # for retrieval weight
    from text2vector.ngrams import Ngrams

    word_vocab_file = "/Users/fanzfeng/DocTemp/ZA/w2v/vocab.txt"
    w2v_file = "/Users/fanzfeng/DocTemp/ZA/w2v/gensim_word2vec2_combine.model"
    assert os.path.exists(word_vocab_file) and os.path.exists(w2v_file)
    engine_file = os.path.join(data_base_path, "w2v_engine.ana")
    data_files = {"train": os.path.join(data_base_path, "retrieval.index"),
                  "eva": os.path.join(data_base_path, "text_label/eva_20191205_.txt")}
    dt_metrics = ["map", "ndcg@3"]

    text_maxlen = 20
    data_refresh = True
    recall_cnt = 10

    se_nlp = ZhNlp(config_lib="jieba", config_dict=word_vocab_file, config_stop=PUNCTS_stopwords, seg_out_list=True)
    # idf = TfIdf(corpus_files=["classify_queries"], ngrams=-1, seg_func=se_nlp.zh_seg, corpus_distinct=True,
    #             base_path=data_base_path, output_model="idf", sparse=False)
    se = SearchEngine(query2rid_file=data_files["train"], cache_file=engine_file,
                      res_col="qid", query_col="question", index_col="seg", vec_model="w2v",
                      model_file=w2v_file, seg_func=se_nlp.zh_seg, weight_func=None)
    if data_refresh:
        se.refresh_index()
    char_vocab_file = os.path.join(data_base_path, " qa_dict.char")
    gram_1 = Ngrams(ngrams=1, vocab_file=char_vocab_file, input_len=text_maxlen)

    # [bert_nlp, gram_1, ][1:]

    sim = SimMatch(seq_len=20, batch_size=16, epoch_num=5, lr=0.001, keep_rate=0.5,
                   files_split=data_files, engine=se, recall_num=recall_cnt,
                   vec_models=[gram_1], match_model="bow", model_path=model_save_path,
                   net_params={"hidden_size": 128, "vocab_size": 264, "emb_size": 64}, eval_metrics=dt_metrics)
    # sim = SimMatch(seq_len=20, batch_size=128, epoch_num=5, lr=0.001, keep_rate=0.7,
    #                files_split=data_files, engine=se, recall_num=recall_cnt,
    #                vec_models=[gram_1], match_model="dssm", model_path=model_save_path, eval_metrics=dt_metrics,
    #                net_params={"hidden_sizes": [512, 128], "vocab_size": 1605, "emb_size": 256})
    # sim = SimMatch(seq_len=20, batch_size=128, epoch_num=5, lr=0.001, keep_rate=0.7,
    #                files_split=data_files, engine=se, recall_num=recall_cnt,
    #                vec_models=[gram_1], match_model="abcnn", model_path=model_save_path, eval_metrics=dt_metrics,
    #                net_params={"model_type": "ABCNN-2", "vocab_size": 1604, "emb_size": 256, "num_layers": 2,
    #                            "num_filters": 50, "window_size": 4, "pool_size": 4})

    sim.refresh(cache=False if data_refresh else True)
    print(sim.process("你是傻子么", "滚蛋"))
