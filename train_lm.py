import os
import tensorflow as tf
import keras
import time
import yaml  # pin install pyyaml
import argparse

from utils_lm import get_data, data_hparams
from transformer import Lm, lm_hparams

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config', type=str, default='config/conf_lm.yaml')
parser.add_argument(
    '-s', '--save_path', type=str, default="save_models/"+time.strftime("%m-%d-%H-%M-%S")+"-logs")
cmd_args = parser.parse_args()

# 自动分配GPU内存，以防止Gpu内存不够的情况
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

def dic2args(dic):
    """
    将dict转换成args的格式。
    """
    params = tf.contrib.training.HParams()
    for keys in dic:
        params.add_hparam(keys, dic[keys])
    return params


# 加载整个训练必要的config
f = open(cmd_args.config, 'r', encoding='utf-8')
parms_dict = yaml.load(f, Loader=yaml.FullLoader)
f.close()

# 训练数据参数
data_args = dic2args(parms_dict['data'])
train_data = get_data(data_args)
batch_num = train_data.batch_num

# 语言模型参数
parms_dict['model']['input_vocab_size'] = len(train_data.pny_vocab)
parms_dict['model']['label_vocab_size'] = len(train_data.han_vocab)
lm_args = dic2args(parms_dict['model'])
lm = Lm(lm_args)

# 训练的参数
epochs = parms_dict['train']['epochs']
save_path = parms_dict['train']['save_path'] = cmd_args.save_path
retrain_dir = parms_dict['train']['retrain_dir']
tensorboard_dir = os.path.join(save_path,"tensorboard")

## 保存pny_vocab、han_bocab、config.yaml、train_log.txt
config_path = os.path.join(save_path,'config')
os.makedirs(config_path,exist_ok=True)
pny_vocab = train_data.pny_vocab
han_vocab = train_data.han_vocab
## 保存pinyinvocab
pny_vocab_file = os.path.join(config_path,"pny_vocab.txt")
f = open(pny_vocab_file,"w",encoding='utf-8')
f.write("\n".join(pny_vocab))
f.close()
## 保存汉字vocab
han_vocab_file = os.path.join(config_path,"han_vocab.txt")
f = open(han_vocab_file,"w",encoding='utf-8')
f.write("\n".join(han_vocab))
f.close()
## 保存config
parms_dict['data']['pny_vocab_file'] = pny_vocab_file
parms_dict['data']['han_vocab_file'] = han_vocab_file
f = open(os.path.join(config_path,"config.yaml"),"w",encoding='utf-8')
yaml.dump(parms_dict,f)
f.close()
## 保存训练log
f = open(os.path.join(config_path,"train_log.txt"),"w")

## 生成saver对象以可以使用 saver.save(sess, checkpointname)生成checkpoint
with lm.graph.as_default():  # tf.Graph().as_default()as_default()
    saver = tf.train.Saver()

with tf.Session(graph=lm.graph) as sess:
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    if retrain_dir:
        # 加载已训练模型
        print('loading language model...')
        latest = tf.train.latest_checkpoint(retrain_dir)
        saver.restore(sess, latest)

    writer = tf.summary.FileWriter(tensorboard_dir, tf.get_default_graph())

    for k in range(epochs):
        # 开始训练
        total_loss = 0
        steps_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost, _ = sess.run([lm.mean_loss, lm.train_op], feed_dict=feed)
            total_loss += cost

            # 显示loss信息
            step = k * batch_num + i
            if step % 100 == 0:
                # 用于tensorboard的loss显示。
                rs = sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, step)
                # 打印训练过程中的loss
                current = time.strftime("%m-%d-%H-%M-%S")
                avg_loss = (total_loss - steps_loss) / 100
                print_s = "[%s INFO] -Training-Epoch-%02d(%.3f), Global Step:%07d, Loss:%.5f, Avgloss:%.5f" %(current, k, i/batch_num, step, cost, avg_loss)
                print(print_s)
                f.write(print_s+"\n")
                steps_loss = total_loss

        # 打印一个epoch的平均loss
        print_s = 'epochs'+str(k+1)+': average loss = '+str(total_loss/batch_num)+"\n"
        print(print_s)
        f.write(print_s)

        # 保存成checkpoint模型
        # if k % 10 == 0:
        saver.save(sess, os.path.join(save_path, 'model_%d' % k))
    saver.save(sess, os.path.join(save_path, 'model_%d' % epochs))
    writer.close()



