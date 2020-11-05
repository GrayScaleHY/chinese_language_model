#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils_lm import decode_ctc, GetEditDistance
from transformer import Lm, lm_hparams
import yaml  # pin install pyyaml
import argparse
import numpy as np
import keras

def dic2args(dic):
    """
    将dict转换成args的格式。
    """
    params = tf.contrib.training.HParams()
    for keys in dic:
        params.add_hparam(keys, dic[keys])
    return params

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config', type=str, default='save_models/11-03-01-49-09-logs/config/config.yaml')
cmd_args = parser.parse_args()

# 自动分配GPU内存，以防止Gpu内存不够的情况
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# 加载整个训练必要的config
f = open(cmd_args.config, 'r', encoding='utf-8')
parms_dict = yaml.load(f, Loader=yaml.FullLoader)
f.close()
lm_args = dic2args(parms_dict['model'])
lm_args.add_hparam('save_path',parms_dict['train']['save_path'])
lm_args.dropout_rate = 0.

### 加载语言模型
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():

    latest = tf.train.latest_checkpoint(lm_args.save_path)
    saver.restore(sess, latest)

## 加载汉字vocab和pinyin vocab。
han_vocab = np.loadtxt(os.path.join(lm_args.save_path,'config','han_vocab.txt'),dtype=str,delimiter="\n")
pny_vocab = np.loadtxt(os.path.join(lm_args.save_path,'config','pny_vocab.txt'),dtype=str,delimiter="\n").tolist()

with sess.as_default():
    # 输入拼音
    text_ = "jin1 tian1 tian1 qi4 ru2 he2"
    # real = ""

    text = text_.strip('\n').split(' ')
    x = np.array([pny_vocab.index(pny) for pny in text])
    x = x.reshape(1, -1)
    preds = sess.run(lm.preds, {lm.x: x})
    got = ''.join(han_vocab[idx] for idx in preds[0])
    # 打印预测结果
    print(text_," --> ",got)

sess.close()
