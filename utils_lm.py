import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm  ## 用于迭代时产生进度条
from scipy.fftpack import fft
from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        text_path='data/text_lm',
        batch_size=2,
        data_length=None)
    return params


class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.text_path = args.text_path
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.pny_vocab_file = args.pny_vocab_file
        self.han_vocab_file = args.han_vocab_file
        self.source_init()

    def source_init(self):
        print('get source list...')
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        
        sub_file = self.text_path
        with open(sub_file, 'r', encoding='utf8') as f:
            data = f.readlines()
        for line in tqdm(data):
            pny, han = line.split('\t')
            self.pny_lst.append(pny.split(' '))
            self.han_lst.append(han.strip('\n'))
        if self.data_length:
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        print('make lm pinyin vocab...',len(self.pny_lst))
        # print(self.pny_lst[:10])
        # print(self.pny_lst[-10:])
        if self.pny_vocab_file:
            self.pny_vocab = np.loadtxt(self.pny_vocab_file, dtype=str).tolist()
        else:
            self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
        print('make lm hanzi vocab...',len(self.han_lst))
        # print(self.han_lst[:10])
        # print(self.han_lst[-10:])
        if self.han_vocab_file:
            self.han_vocab = np.loadtxt(self.han_vocab_file,dtype=str).tolist()
        else:
            self.han_vocab = self.mk_lm_han_vocab(self.han_lst)
        self.batch_num = len(self.pny_lst) // self.batch_size # 分多少个batch


    def get_lm_batch(self):
        """
        将多组数据打包成batch。
        """
        
        for i in range(self.batch_num):
            begin = i * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])

            input_batch = np.array(
                [self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])

            ## yield 迭代器，在外部用next调用。
            yield input_batch, label_batch

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

# word error rate------------------------------------
def GetEditDistance(str1, str2):
	leven_cost = 0
	s = difflib.SequenceMatcher(None, str1, str2)
	for tag, i1, i2, j1, j2 in s.get_opcodes():
		if tag == 'replace':
			leven_cost += max(i2-i1, j2-j1)
		elif tag == 'insert':
			leven_cost += (j2-j1)
		elif tag == 'delete':
			leven_cost += (i2-i1)
	return leven_cost

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
	result = num_result[:, :, :]
	in_len = np.zeros((1), dtype = np.int32)
	in_len[0] = result.shape[1]
	r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
	r1 = K.get_value(r[0][0])
	r1 = r1[0]
	text = []
	for i in r1:
		text.append(num2word[i])
	return r1, text
