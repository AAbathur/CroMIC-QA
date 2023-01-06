import os
import pickle
import tensorflow as tf 
import array
from tqdm import tqdm
import six
from collections import Counter
import collections

def get_vocab(vocab_path):
    # 从config.vocab_path=r"data\WV\vocab_vec_200d.txt"读取文件,获得词表
    vocab = []
    with open(vocab_path,'r',encoding='utf-8') as fv:
        vocab = [line.strip().split(' ',1)[0] for line in fv]
    return vocab[1:]

def get_vocab_list(vocab_path):
    vocab_list = []
    with open(vocab_path,'r',encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            if len(line) == 0: continue
            line_list = line.split('#')
            vocab_list.append(line_list[0])
    print("vocab_list done!!!")
    return vocab_list
def get_vocab_dict(vocab_path):
    vocab_dict = collections.OrderedDict()
    with open(vocab_path,'r',encoding='utf-8') as f1:
        for line in f1.readlines():
            line = line.strip()
            if len(line) == 0: continue
            line_list = line.split('#')
            for word in line_list:
                vocab_dict[word] = line_list[0]
    print("vocab_dict done!!!")
    return vocab_dict

def get_vocab_idx_dict(vocab_path):
    word2idx = {}
    with open(vocab_path,'r',encoding='utf-8') as f1:
        for i,line in enumerate(f1):
            line = line.strip()
            line_list = line.split('#')
            word2idx[line_list[0]] = i
    print("get_vocab_idx_dict done!!!")
    return word2idx

def load_word_vectors(root=r"data\WV",wv_type=r'vocab_vec',dim=200):
    "读取txt文件,写入pk文件中. 返回值(wv_dict,wv_arr,wv_size),wv_dict是每个词对应的index,wv_ar每一行是对应词向量,wv_size是词向量的长度"
    if isinstance(dim,int):
        dim = str(dim) + 'd'
    fname = os.path.join(root,wv_type+'_'+dim)
    print('fname is ',fname)

    if os.path.isfile(fname+'.pk'):
        fname_pk = fname + '.pk'
        print('loading word vectors from', fname_pk)
        f = open(fname_pk,'rb')
        ret = pickle.load(f)
        f.close()
        return ret

    if os.path.isfile(fname+'.txt'):
        fname_txt = fname+'.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), 200
    if cm is not None:
        print
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            #第一行是词向量文件中:词数量 词向量维度
            if line == 0: continue
    
            entries = cm[line].strip().split(b' ')
            word, vector = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(vector)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF-8 token',repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in vector)
            wv_tokens.append(word)
        
    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = tf.reshape(tf.convert_to_tensor(wv_arr), [-1,wv_size])
    ret = (wv_dict, wv_arr, wv_size)

    f = open(fname+'.pk','wb')
    pickle.dump(ret,f)
    f.close()
    return ret


def vocab_cover():
    # 统计30.4w中高频词在181431个词中的覆盖率
    real_vocab = r"data\real_vocab.txt"
    real_vocab_list = []
    with open(real_vocab,'r',encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[0]
            real_vocab_list.append(word)
    print('real_vocab_list:', real_vocab_list[:10])
    all_vocab_dict = {}
    with open(r"data\vocab.txt",'r',encoding='utf-8') as f2:
        for i, line in enumerate(f2):
            line_list = line.rstrip().split('\t')
            if i == 268:
                print(line)
                print(line_list)
            word = line_list[0]
            tf = int(line_list[1])
            all_vocab_dict[word] = tf
            
    total = 0
    count = 0
    for key, value in all_vocab_dict.items():
        if value >= 100:
            total += 1
            if key in real_vocab_list:
                count += 1
    print('total: ',total)
    print('count: ',count)
    print('percent: ',count/total)   

def user_dict_cover_vocab():
    real_vocab = r"data\real_vocab.txt"
    real_vocab_list = []
    with open(real_vocab,'r',encoding='utf-8') as f1:
        for line in f1:
            word = line.strip().split('\t')[0]
            real_vocab_list.append(word)
    print('real_vocab_list:', real_vocab_list[:10])
    user_list = []
    with open(r"data\user_dict.txt", 'r', encoding='utf-8') as f2:
        for line in f2:
            word = line.strip()
            user_list.append(word)
    user_list = list(set(user_list))
    uncover_list = []
    total = 0
    count = 0
    for word in user_list:
        total += 1
        if word in real_vocab_list:
            count += 1
        else:
            uncover_list.append(word)
    print('total: ',total)
    print('count: ', count)
    print('percent: ', count/total)
    print(uncover_list)
    with open(r"data\uncover_user_dict.txt",'w',encoding='utf-8') as f3:
        for word in uncover_list:
            f3.write(word+'\n')

if __name__ == "__main__":
    #vocab_cover()
    user_dict_cover_vocab()
