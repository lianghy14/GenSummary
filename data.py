# coding: utf-8
import json
import collections
import re
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

# file path
raw_data = 'sample-1M.jsonl'
train_article = 'train/train_article.src'
train_title = 'train/train_title.src'
test_article = 'train/test_article.src'
test_title = 'train/test_title.src'

# data size
train_num = 300000
test_num = 10000


def clean_str(sentence):
    sentence = re.sub("[•£™�€âÂ®¢❤»©# '\xa0' '\ufe0f' '\xad' '\u200b' '\n' '\t' '\r' '\b' '\000' '\v' '\f' 每]+", " ", sentence)
    sentence = re.sub("[’‘]+","'",sentence)
    sentence = re.sub("[”“]+","\"",sentence)
    sentence = re.sub("\xa0",u' ',sentence)
    return sentence

def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def build_dict(step):
    if step == "train":
        train_article_list = get_text_list(train_article)
        train_title_list = get_text_list(train_title)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)
        
        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("train/word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("train/word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    article_max_len = 300
    summary_max_len = 50

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))
        
    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article)
        title_list = get_text_list(train_title)
    elif step == "valid":
        article_list = get_text_list(test_article)
        title_list = get_text_list(test_title)
    else:
        raise NotImplementedError

    x = list(map(lambda d: word_tokenize(d), article_list))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:article_max_len], x))
    x = list(map(lambda d: d + (article_max_len - len(d)) * [word_dict["<padding>"]], x))

    y = list(map(lambda d: word_tokenize(d), title_list))
    y = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), y))
    y = list(map(lambda d: d[:(summary_max_len-1)], y))

    return x, y


def get_text_list(data_path):
    with open(data_path, "r") as f:
        return list(f.readlines())

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]



def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)


# generate data set
if __name__ == '__main__':
    with open(raw_data,'r') as f:
        lines_written = 0
        lines_failed = 0
        for i in range(train_num):
            tmp_str = f.readline()
            tmp = json.loads(tmp_str)
            
            test = tmp['title']
            test_len = len(word_tokenize(test))
            if is_chinese(test) or test_len < 2 or test_len > 50:
                lines_failed += 1
                continue
                
            test = tmp['content']
            test_len = len(word_tokenize(test))
            if test_len < 100 or test_len > 300:
                lines_failed += 1
                continue
            
            line = ''
            try:  
                with open('Debug.txt','w') as file:
                    file.write(clean_str(tmp['title']))
                with open(train_article,'a') as article:
                    line = clean_str(tmp['content'])            
                    article.write(line)
                    article.write('\n')
                with open(train_title,'a') as title:
                    line = clean_str(tmp['title'])
                    title.write(line)
                    title.write('\n')
                lines_written += 1
            except:
                lines_failed += 1

        print(lines_written)
        print(lines_failed)
        
        lines_written = 0
        lines_failed = 0        
        for i in range(test_num):
            tmp_str = f.readline()
            tmp = json.loads(tmp_str)
            
            test = tmp['title']
            if is_chinese(test) or len(word_tokenize(test)) < 2 or len(word_tokenize(test)) > 50:
                lines_failed += 1
                continue
            test = tmp['content']
            if len(word_tokenize(test)) < 100 or len(word_tokenize(test)) > 300:
                lines_failed += 1
                continue

            line = ''
            try: 
                with open('Debug.txt','w') as file:
                    file.write(clean_str(tmp['title']))
                with open(test_article,'a') as article:
                    line = clean_str(tmp['content'])            
                    article.write(line)
                    article.write('\n')
                with open(test_title,'a') as title:
                    line = clean_str(tmp['title'])
                    title.write(line)
                    title.write('\n')
                    lines_written += 1
            except:
                lines_failed += 1

        print(lines_written)
        print(lines_failed)