# -*- coding: utf-8 -*-
# version=3.6.4
# @Author  : fanzfeng
'''
通过文本分类解决问题文本的匹配问题
'''
import pandas as pd
import sys, os, time, platform
import json

botPath = "/".join(os.path.split(os.path.realpath(__file__))[0].split('/')[:-1])
print(botPath)
sys.path.append(botPath)
from nets.network import nets_dict, tf, np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

root_path = botPath
data_base_path = os.path.join(root_path, "data/service")
data_model_path = os.path.join(data_base_path, "text_label")
model_dir = os.path.join(root_path, "models")
model_save_path = model_dir+"/{m}_classify"
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


class TextClassifier(object):
    def __init__(self, emb_size, hidden_size, num_class, batch_size, epoch_num, lr, keep_rate,
                 vocab_size=None, seq_len=None, samples_num=None, model_path=model_save_path, origin_file="",
                 train_test_files=None, model_data_path=data_model_path,
                 labels_conf=os.path.join(data_base_path, "labels.json"), vec_models=[], model_name="bert"):
        self.r_state = 666
        self.origin_file = origin_file
        self.train_test_files = train_test_files
        self.vec_models = vec_models
        self.model_name = model_name
        assert os.path.exists(self.origin_file) or self.train_test_files is not None
        assert self.model_name in nets_dict
        assert isinstance(self.vec_models, list) and len(self.vec_models) > 0
        self.model_data_path = model_data_path
        self.model_path = model_path.format(m=model_name)

        self.featurizers = []
        for m in vec_models:
            if isinstance(m, list):
                self.featurizers += m
            else:
                self.featurizers.append(m)
        self.num_feed_x = len(self.featurizers)*1

        self.samples_num = samples_num
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_class = num_class

        self.hidden_size = hidden_size
        self.keep_rate = keep_rate
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.lr = lr

        self.sep = "\t"
        self.sample_dist = 1.0
        self.train_rate = 0.7
        self.query2qid = None
        self.train_num, self.eva_num = None, None
        self.cols = ["text", "label"]
        self.y_col = self.cols[-1]
        self.text_col = self.cols[0]
        self.label2id, self.id2label = {}, {}
        self.labels_file = labels_conf

        for f_path in [self.model_data_path, self.model_path]:
            if not os.path.exists(f_path):
                os.mkdir(f_path)
        if self.train_test_files:
            self.train_file, self.eva_file = self.train_test_files["train"], self.train_test_files["eva"]

        # if not os.path.exists(self.train_file) or not os.path.exists(self.eva_file):
        #     self.gen_data(origin_file=origin_file)

        if not os.path.exists(self.labels_file):
            self.gen_data(origin_file=origin_file)
        else:
            print("!! read id2label from json file")
            with open(self.labels_file, "r", encoding="utf-8") as fp:
                self.id2label = json.load(fp)
                self.label2id = {y: int(k) for k, y in self.id2label.items()}

        self.tf_dtypes, self.np_dtypes = [], []
        for m in self.featurizers:
            self.tf_dtypes += [tf.float32 if len(m.output_shape) > 2 else tf.int32]*1
            self.np_dtypes += [np.float32 if len(m.output_shape) > 2 else np.int32]*1
        self.tf_dtypes.append(tf.int32)
        self.np_dtypes.append(np.int32)
        self.input_tensors = []

        self.test_y, self.keep_prob = None, None
        print("Initializer network...")
        self.define_tensor()
        self.net_loss, self.one_hot_labels, self.pred_prob = None, None, None
        self.net_init()

        self.session = None
        model_save = tf.train.get_checkpoint_state(self.model_path)
        if model_save and model_save.model_checkpoint_path:
            print("Loading {} model...".format(self.model_name))
            # tf.reset_default_graph()
            self.session = tf.Session()
            self.session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            try:
                saver.restore(self.session, model_save.model_checkpoint_path)
                print("Classify model is ready")
            except:
                print("Load classify model Failed !!")
        else:
            print("Classify model not exists")

    def define_tensor(self):
        k = 0
        for i in range(len(self.featurizers)):
            for j in range(1):
                self.input_tensors.append(tf.placeholder(self.tf_dtypes[k], self.featurizers[i].output_shape,
                                                         name="input_{}_{}".format(i, j)))
            k += 1
        self.test_y = tf.placeholder(self.tf_dtypes[-1], [None, self.num_class], name="input_label")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def net_init(self):
        tensor_dict = {"input": self.input_tensors, "output": self.test_y, "keep_rate": self.keep_prob}
        net_conf = {"emb_size": self.emb_size, "input_len": self.seq_len, "vocab_size": self.vocab_size,
                    "num_class": self.num_class, "hidden_size": self.hidden_size}
        net_sim = nets_dict[self.model_name](conf=net_conf)
        if net_sim:
            self.pred_prob, self.one_hot_labels, self.net_loss = net_sim.net(tensor_dict)

    @staticmethod
    def under_sample(data, label_col, sample_dist, r_state):
        if label_col in data.columns and data[label_col].value_counts().shape[0] == 2:
            pos_num, neg_num = sum(data[label_col] == 1), sum(data[label_col] == 0)
            if neg_num > pos_num:
                sample_num = int(min(neg_num, sample_dist*pos_num))
                neg_df = data[data[label_col] == 0].sample(n=sample_num, random_state=r_state)
                pos_df = data[data[label_col] == 1]
            else:
                sample_num = int(min(pos_num, sample_dist * neg_num))
                pos_df = data[data["y"] == 1].sample(n=sample_num, random_state=r_state)
                neg_df = data[data["y"] == 0]
            new_df = pd.concat([neg_df, pos_df], ignore_index=True)
            return new_df

    @staticmethod
    def label_map(v, label2ix):
        ij = label2ix.get(v)
        if ij is None:
            print("irlegal label:", v)
        tmp_v = [0] * len(label2ix)
        if v in label2ix:
            tmp_v[ij] += 1
            return tmp_v

    def gen_data(self, origin_file, r_state=666):
        print("\nsplit index data, get train and eva data")
        db = pd.read_csv(origin_file, sep=self.sep)#.drop_duplicates()
        db.columns = self.cols
        class_list = sorted([c for c in set(db[self.y_col])])
        self.num_class = len(class_list)
        print("labels cnt: {} >>> {}".format(self.num_class, class_list))
        self.label2id = {v: i for i, v in enumerate(class_list)}
        self.id2label = {i: v for i, v in enumerate(class_list)}
        with open(self.labels_file, "w", encoding="utf-8") as fp:
            json.dump(self.id2label, fp, indent=4, ensure_ascii=False)
        print("origin label dist: ", db[self.y_col].value_counts().head())
        train_df = db.sample(frac=self.train_rate, random_state=r_state)
        eva_df = db[~db.index.isin(train_df.index)]
        if self.num_class == 2 and self.sample_dist > 0:
            train_data = self.under_sample(data=train_df, label_col=self.y_col, sample_dist=self.sample_dist,
                                           r_state=r_state)
        else:
            train_data = train_df.drop_duplicates()
        print("train_data label dist: ", train_data[self.y_col].value_counts(normalize=True).head())

        self.train_num, self.eva_num = len(train_data), len(eva_df)
        print("samples dist: train {} eva {}".format(self.train_num, self.eva_num))
        train_data.to_csv(self.train_file, sep=self.sep, index=False, header=False)
        eva_df.to_csv(self.eva_file, sep=self.sep, index=False, header=False)
        print("data for classifier network ready")
        return {"train": train_data, "eva": eva_df}

    def featurizer(self, t2v, text_list):
        if t2v:
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

    def load_data(self, data_file, pkl_file, header=False, cache=True):
        # read data/featurizer
        if cache and os.path.exists(pkl_file):
            full_df = pd.read_pickle(pkl_file)
            print("Load feature data {}".format(pkl_file))
        else:
            print("\nLoad data file <{}>".format(data_file))
            if header:
                df = pd.read_csv(data_file, sep=self.sep).dropna()
                df.columns = self.cols
            else:
                df = pd.read_csv(data_file, sep=self.sep, header=None, names=self.cols).dropna()
            if "train" in data_file:
                df = df.drop_duplicates()
                class_list = sorted([c for c in set(df[self.y_col])])
                self.num_class = len(class_list)
                print("labels cnt: {} >>> {}".format(self.num_class, class_list))
                self.label2id = {v: i for i, v in enumerate(class_list)}
                self.id2label = {i: v for i, v in enumerate(class_list)}
                with open(self.labels_file, "w", encoding="utf-8") as fp:
                    print("!! refresh id2label json file")
                    json.dump(self.id2label, fp, indent=4, ensure_ascii=False)
            print("   label dist:\n ", df[self.y_col].value_counts().head())
            full_df = df.copy()
            full_df["y_vec"] = full_df[self.y_col].apply(lambda y: self.label_map(y, self.label2id))
            c, j = self.text_col, 0
            for i in range(len(self.featurizers)):
                if self.featurizers[i].name == "bert" and self.model_name == "bert":
                    full_df["full_text"] = full_df.apply(
                        lambda sr: "[sep]".join([str(s) for s in sr[[self.text_col]].tolist()]), axis=1)
                    full_df["vec"] = self.featurizer(self.featurizers[i], full_df["full_text"].tolist()).tolist()
                if self.featurizers[i].name == "bert" and self.model_name == "bert":
                    full_df["vec_{}_{}".format(i, j)] = full_df["vec"]
                else:
                    full_df["vec_{}_{}".format(i, j)] = self.featurizer(self.featurizers[i], full_df[c].tolist()).tolist()
            full_df.to_pickle(pkl_file)
        return full_df

    def batch_iter(self, data, eva=False):
        if eva:
            df = data.copy()
            x_vec_list = []
            j = 0
            for i in range(len(self.featurizers)):
                x_vec_list += [np.array(df["vec_{}_{}".format(i, j)].tolist())]
            yield x_vec_list+[np.array(df["y_vec"].tolist())]
        else:
            df = data.sample(frac=1.0, random_state=self.r_state)
            obs, j = len(df), 0
            batch_num = int(obs/self.batch_size)
            data_feed = []
            for i in range(len(self.featurizers)):
                data_feed += [np.array([df["vec_{}_{}".format(i, j)].iloc[ix] for ix in range(obs)])]
            data_feed += [np.array(df["y_vec"].tolist())]
            for jk in range(batch_num):
                yield [v[jk*self.batch_size:min(obs, (jk+1)*self.batch_size)] for v in data_feed]

    def train(self, cache=True, max_models=7):
        # data/featurizer--->model
        if self.train_test_files is not None:
            print("\nGet data with feature...")
            data_full = {k: self.load_data(data_file=f, pkl_file=os.path.join(self.model_data_path, k+".pkl"),
                                           cache=cache) for k, f in self.train_test_files.items()}
            print("  queries data: {}".format("\t".join(['%s=%d' % (k, len(v)) for k, v in data_full.items()])))

        pred_index = tf.argmax(self.pred_prob, 1)
        correct_pred = tf.equal(pred_index, tf.argmax(self.one_hot_labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_pred, "float"))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.net_loss)

        print("Begin Training...")
        saver = tf.train.Saver(max_to_keep=max_models)
        self.session = tf.Session()
        j = 0
        self.session.run(tf.global_variables_initializer())
        for e in range(self.epoch_num):
            for input_all in self.batch_iter(data_full["train"], eva=False):
                feeds = {self.keep_prob: self.keep_rate, self.test_y: input_all[-1].astype(self.np_dtypes[-1])}
                for k in range(self.num_feed_x):
                    feeds.update({self.input_tensors[k]: input_all[k].astype(self.np_dtypes[k])})
                cost, _, train_acc = self.session.run([self.net_loss, optimizer, acc], feed_dict=feeds)
                j += 1
                if j % 10 == 0:
                    print(" " * 4, ">>> train loss: {} accuracy: {}".format(cost, train_acc))
            for eva_input in self.batch_iter(data_full["eva"], eva=True):
                eva_feeds = {self.keep_prob: 1.0, self.test_y: eva_input[-1].astype(self.np_dtypes[-1])}
                for k in range(self.num_feed_x):
                    eva_feeds[self.input_tensors[k]] = eva_input[k].astype(self.np_dtypes[k])
                eva_loss, accuracy = self.session.run([self.net_loss, acc], feed_dict=eva_feeds)
            print("epoch %d loss on validation data: %f | accuracy--> : %f" % (e+1, eva_loss, accuracy))
            saver.save(self.session, self.model_path + "/model_e{}.ckpt".format(e+1))
        print("Train Finished")

    # @timer(prefix="api")
    def process(self, text):
        if self.session is not None and isinstance(text, str) and len(text) > 0:
            p_inputs = []
            for i in range(len(self.featurizers)):
                shapes = self.featurizers[i].output_shape
                shapes = [1] + shapes[1:]
                if self.featurizers[i].name == "bert" and self.model_name == "bert":
                    vec = np.array(self.featurizers[i].text2vec(text))
                    if self.featurizers[i].name == "bert" and self.model_name == "bert":
                        p_inputs.append(vec.reshape(shapes))
                else:
                    p_inputs.append(self.featurizer(self.featurizers[i], [text]).reshape(shapes))
            p_feeds = {self.keep_prob: 1.0,
                       self.test_y: np.array([0]*self.num_class).reshape(1, self.num_class).astype(self.np_dtypes[-1])}
            for j in range(self.num_feed_x):
                p_feeds[self.input_tensors[j]] = p_inputs[j].astype(self.np_dtypes[j])
            pred_dist = self.session.run(self.pred_prob, feed_dict=p_feeds)
            ix = np.argmax(pred_dist)
            try:
                return self.id2label[ix], pred_dist
            except:
                return self.id2label[str(ix)], pred_dist

    def refresh(self, cache=True):
        print("Refreshing data and model...")
        # self.gen_data(origin_file=self.origin_file)
        self.train(cache=cache)
        print("Refresh Finished")


if __name__ == "__main__":
    from text2vector.vector_model import Text2Vector
    from text2vector.ngrams import Ngrams

    char_vocab_file = os.path.join(data_base_path, "qa_dict.char")
    text_maxlen = 20
    data_refresh = True
    data_key = "20191205"
    # t2v = Text2Vector(w2v_file="bert_api")
    gram_1 = Ngrams(ngrams=1, vocab_file=char_vocab_file, input_len=text_maxlen)
    data_files_split = {"train": os.path.join(data_model_path, "train_{}.txt".format(data_key)),
                        "eva": os.path.join(data_model_path, "eva_{}_.txt".format(data_key))}
    # sim = TextClassifier(emb_size=768, hidden_size=1024, num_class=35, batch_size=128, epoch_num=500,
    #                      lr=0.001, keep_rate=0.6,model_type="bert",
    #                      origin_file=os.path.join(data_base_path, "classify_queries"),
    #                      model_path=model_save_path,
    #                      model_data_path=os.path.join(data_base_path, "text_label"),
    #                      labels_conf=os.path.join(data_base_path, "labels.json"),
    #                      samples_num=None, force_model=True)
    sim = TextClassifier(seq_len=text_maxlen, emb_size=64, hidden_size=None, num_class=5, batch_size=16, epoch_num=5,
                         lr=0.001, keep_rate=0.6, vocab_size=264,
                         origin_file=os.path.join(data_base_path, "classify_queries"),
                         model_path=model_save_path,
                         train_test_files=data_files_split,
                         model_data_path=os.path.join(data_base_path, "text_label"),
                         labels_conf=os.path.join(data_base_path, "labels.json"),
                         samples_num=None, model_name="textcnn", vec_models=[gram_1, ])
    sim.refresh(cache=False if data_refresh else True)
    print(sim.process("我想知道一下，能告诉我么"))
