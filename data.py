# dataset for answer re-ranking model
import os
import time
import config
import utils
import json
import jieba
import numpy as np 
import array
import csv
#jieba.load_userdict(config.user_dict)
### 加载停用词表,主要是处理特殊字符
import tensorflow as tf
#from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
import collections
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt 
import random

class VQADS():
    def __init__(self):
        super(VQADS,self).__init__()
        self.data_path = config.data_path
        self.image_url = config.image_url
        self.vocab = utils.get_vocab(config.vocab_path)
        print("len of vocab is",len(self.vocab))
        # 词的dict, key是token, value是idx. 最后放一个UN表示unknow的
        self.token_to_idx = utils.load_word_vectors()[0]
        print("len of token_to_idx is",len(self.token_to_idx))
        # 一个句子(包括question和answer)的最大长度
        self.max_sentence_length = config.max_q_length

    def get_questions(self):
        self.questions = list(prepare_questions(self.data_path, self.vocab))
        print(len(self.questions))
        self.questions = [self._encode_question(q) for q in self.questions]
        return self.questions

    def get_answers(self):
        self.answers = list(prepare_answer(self.data_path, self.vocab))
        print(len(self.answers))
        self.answers = [self._encode_answers(a) for a in self.answers]
        return self.answers

    def get_images(self):
        self.images = list(prepare_image(self.image_url))
        return self.images

    def get_scores(self):
        self.scores = list(prepare_rel_score(self.data_path))
        return self.scores

    @property
    def max_question_length(self):
        # 所有question中最长的token_list长度
        if not hasattr(self, '_max_length'):
            if not hasattr(self, 'questions'):
                data_max_length = 26
            else:
                data_max_length = max(map(len, self.questions))
            self._max_length = min(config.max_q_length, data_max_length)
        return self._max_length

    @property
    def num_token(self):
        return len(self.token_to_idx)

    def _encode_question(self, question):
        vec = np.full(self.max_sentence_length, self.num_token)
        for i, token in enumerate(question):
            if i >= self.max_sentence_length:
                break
            index = self.token_to_idx.get(token, self.num_token)
            vec[i] = index        
        return vec, min(len(question), self.max_sentence_length)

    def _encode_answers(self, answers):
        return [self._encode_question(answer) for answer in answers]

    def _load_image(self,item):
        image_urls = r"data\image_name_urls.txt"
        with open(image_urls,'r',encoding='utf-8') as f1:
            all_lines = f1.readlines()
            line_list = all_lines[item].strip().split('\t')
            images = [a.split('#')[0] for a in line_list[2:] if len(a.split('#'))==2 ]
        return images
    
    def __getitem__(self, item):
        q, q_len = self.questions[item]
        answers = self.answers[item]
        images = self._load_image(item)
        scores = self.scores[item]
        return q, q_len, answers, images, scores 


def prepare_questions(data_path,vocab):
    jieba.load_userdict(config.user_dict)
    with open(data_path,'r',encoding='utf-8') as f1:
        flag = 0
        for line in f1.readlines():
            if flag%10000==0:
                print(flag)
            flag += 1
            line_dict = json.loads(line)
            question = line_dict['question']['q_content']
            yield [i for i in jieba.cut(question) if i in vocab]

def prepare_answer(data_path,vocab):
    jieba.load_userdict(config.user_dict)
    def cut_answer(s):
        return [i for i in jieba.cut(s) if i in vocab]
    with open(data_path,'r',encoding='utf-8') as f2:
        flag = 0
        for line in f2.readlines():
            if flag%10000==0:
                print(flag)
            flag += 1
            line_dict = json.loads(line)
            reply_list = line_dict['replies']
            answers = [reply['reply_content'].strip() for reply in reply_list ]
            yield list(map(cut_answer,answers))

def prepare_rel_score(data_path):
    with open(data_path,'r',encoding='utf-8') as f3:
        for line in f3:
            line_dict = json.loads(line)
            reply_list = line_dict['replies']
            yield [(reply['zancount'], reply['isgood'], reply['reply_level']) for reply in reply_list]

def encoded_question_to_file(file_path, questions):
    with open(file_path,'w',encoding='utf-8') as fq:
        for q, q_len in questions:
            print(q.shape)
            q_list = q.tolist()[:q_len]
            q_str = list(map(str,q_list))
            q_str = ' '.join(q_str)
            fq.write(str(q_len)+'\t'+q_str+'\n')

def encoded_answer_to_file(file_path, answers):
    with open(file_path,'w',encoding='utf-8') as fa:
        for answer in answers:
            for a, a_len in answer:
                a_list = a.tolist()[:a_len]
                a_str = list(map(str,a_list))
                a_str = ' '.join(a_str)
                fa.write(str(a_len)+'\t'+a_str+'#')
            fa.write('\n')

def get_part_data(in_path, out_path, index_path):
    index_list = []
    with open(index_path,'r',encoding='utf-8') as fi:
        reader = csv.reader(fi)
        for line in reader:
            idx = int(line[2][:-4])
            index_list.append(idx)

    with open(in_path,'r',encoding='utf-8') as f1, open(out_path,'w',encoding='utf-8') as f2:
        all_lines = f1.readlines()
        for i,idx in enumerate(index_list):
            print(i)
            line = all_lines[idx]
            f2.write(str(idx)+'\t'+line)

def get_relevance_score(raw_data, index_file, out_path):
    ### 根据index_file=r"data\part_encoded_question.txt"读取raw_data对应行的reply的zancount,is_good,replier_level.
    ### 同一行的reply格式: 行号 \t #zancount is_good replier_level#zancount is_good replier_level#...
    index_list = []
    with open(index_file,'r',encoding='utf-8') as f1:
        for line in f1:
            line_list = line.strip().split('\t',1)
            index_list.append(int(line_list[0]))
    with open(raw_data,'r',encoding='utf-8') as f2, open(out_path,'w',encoding='utf-8') as f3:
        all_line = f2.readlines()
        for i,idx in enumerate(index_list):
            print(i)
            line = all_line[idx]
            line_dict = json.loads(line)
            reply_list = line_dict['replies']
            score_list = []
            for reply in reply_list:
                a = reply['zancount']
                b = reply['is_good']
                c = reply['replier_level']
                score_str = str(a)+' '+str(b)+' '+str(c)
                score_list.append(score_str)
            s_str = '#'.join(score_list)
            f3.write(str(idx)+'\t'+s_str+'\n')
    
def pad_func(data_list, length):
    if len(data_list) == 0:
        x = random.randint(0,181430)
        data_list.append(x)
    if len(data_list)>length:
        return data_list[:length]
    else:
        pad_list = [181431 for _ in range(length-len(data_list))]
        return data_list + pad_list

def get_question_ds(file_path="data/part_encoded_question.txt",length=30):
    # 从file_path读取编码后的question,进行pad操作后返回数据集,length为设定的长度
    questions_list = []
    idxs = []
    with open(file_path,'r',encoding='utf-8') as f1:
        for i,line in enumerate(f1):
            #if i>=100: break
            line_list = line.strip().split('\t')
            idxs.append(line_list[0])
            encode_q_list = line_list[-1].split(' ')
            q_list = list(map(int, encode_q_list))
            padded_q_list = pad_func(q_list,length)
            questions_list.append(padded_q_list)
    return questions_list,idxs

def compute_label(score_str):
    ## score_str: "zancount is_good replier_level"
    s_list = score_str.split(' ')
    s_list = list(map(int,s_list))
    label = 5 + s_list[0]*1 + s_list[1]*50 + s_list[2]*5
    if label > 125:
        label = 125
    return label

def get_answer_ds(answer_path="data/part_encoded_answer.txt",label_path="data/part_label.txt",ans_num=4,length=50):
    # 读取file_path中的answer数据,ans_num指定每个问题对应的答案个数,length控制每句话长度
    all_answers = []
    all_labels = []
    flag = 0
    abandon_idxs = []
    with open(answer_path,'r',encoding='utf-8') as f2, open(label_path,'r',encoding='utf-8') as f3:
        all_ans_path = f2.readlines()
        all_label_path = f3.readlines()
        for i in range(len(all_ans_path)):
            #if i>=100: break
            #if i%5000==0: print(i)
            ans_line = all_ans_path[i]
            score_line = all_label_path[i]
            score_list = score_line.strip().split('\t',1)[1]
            score_list = score_list.split('#')
            label_list = list(map(compute_label,score_list))
                
            answers_list = ans_line.strip().split('\t',1)[1]
            answers_list = answers_list.split('#')[:-1]
            encoded_answers = []
            empty_idx = [] #记录此条数据中n个答案中长度为0的
            for j,ans in enumerate(answers_list):
                ans_len = int(ans.split('\t')[0])
                ans = ans.split('\t')[1]
                ans_list = ans.split(' ')
                ans_list = list(filter(lambda x: x!='',ans_list))
                ans_list = list(map(int,ans_list))
                if len(ans_list):
                    padded_ans_list = pad_func(ans_list,length)
                    encoded_answers.append(padded_ans_list)
                else:
                    empty_idx.append(j)
            empty_idx.reverse()
            for idx in empty_idx:
                label_list.pop(idx)
            """if len(score_list) != len(label_list):
                print('line num is {}'.format(i))
                print(len(score_list))
                print(len(label_list))"""
            assert len(label_list) == len(encoded_answers)
            # 对于label_list长度为0时,跳过
            if len(label_list) == 0: 
                abandon_idxs.append(i)
                flag += 1
                continue
            # 根据label值选前ans_num个,不足的补全
            z = list(zip(encoded_answers,label_list))
            #sort的目的是舍弃部分答案时按照分数大小进行取舍;update:舍弃应该更加随机
            #sorted_z = sorted(z, key=lambda x: (x[1],len(x[0])),reverse=True)
            random.seed(10)
            if len(z) >= ans_num:
                random.shuffle(z)
                a,b = zip(*z)
                all_answers.append(a[:ans_num])
                all_labels.append(b[:ans_num])
            else:
                x = [181431 for _ in range(50)]
                len_z = len(z)
                for i in range(ans_num - len_z):
                    z.append((x,0))
                random.shuffle(z)
                a,b = zip(*z)
                all_answers.append(a)
                all_labels.append(b)
    print("total abandon num is {}".format(flag))
    return all_answers, all_labels,abandon_idxs


def image_vec_split(random_seed,img_vec_type,data_type,abandon_idxs=[],train_val_test=[0.85,0.10,0.05]):
    # all_vec: origin file of all image vec to split
    # random_seed: random_seed to match with random questions
    # img_vec_type: ['MN','CDR']
    # train_val_test: list of split percent,[train_percent,val_percent,test_percent]
    # data_type: int, [1,2,3]
    print('calling image_vec_split func,img_vec_type:{}, data_type:{}'.format(img_vec_type, data_type))
    if data_type == 1:
        folder_path = "data/type1_useless_questions/"
    if data_type == 2:
        folder_path = "data/type2_crop_only_questions/"
    if data_type == 3:
        folder_path = ""
    all_vec_path = folder_path + img_vec_type + '_type'+str(data_type)+'_image_vec_1280d.npy'
    train_path = folder_path + img_vec_type + '_train_images_vec_'+str(random_seed)+'.npy'
    val_path = folder_path + img_vec_type + '_val_images_vec_'+str(random_seed)+'.npy'
    test_path = folder_path + img_vec_type + '_test_images_vec_'+str(random_seed)+'.npy'
    
    all_vec = np.load(all_vec_path)
    print('origin all_vec shape: ', all_vec.shape)
    abandon_idxs = sorted(abandon_idxs,reverse=True)
    for i in abandon_idxs:
        all_vec = np.delete(all_vec,i,0)
    print('after del abandon idxs, all_vec shape: ',all_vec.shape)

    total_num = all_vec.shape[0]
    all_vec = shuffle(all_vec,random_state=random_seed)

    x1 = all_vec[:int(train_val_test[0]*total_num)]
    print('train vec shape: ',x1.shape)
    print('train path: ',train_path)
    np.save(train_path,x1)

    x1 = all_vec[int(train_val_test[0]*total_num):int((train_val_test[0]+train_val_test[1])*total_num)]
    print('val vec shape: ',x1.shape)
    print('val path: ', val_path)
    np.save(val_path,x1)

    x1 = all_vec[int((train_val_test[0]+train_val_test[1])*total_num):]
    print('test vec shape: ',x1.shape)
    print('test path: ',test_path)
    np.save(test_path,x1)

def rank_pos_ans_ds(pos_file, pos_num):

    def pad_none_sentence(ans,label,num):
        assert len(ans) <= num, 'ans num greater than num: {}'.format(num)
        padding_sentence = pad_func([],50)
        for _ in range(num - len(ans)):
            ans.append(padding_sentence)
            label.append(0)
        return ans,label

    pos_answers = []
    pos_label = []
    with open(pos_file,'r',encoding='utf-8') as f1:
        tmp_ans = []
        tmp_label = []
        tmp_qid= ''
        for i, line in enumerate(f1):
            line_list = line.strip().split('\t')
            qid = line_list[0]
            flag = 0
            if len(line_list) == 5:
                encode_tokens = []
                flag = 1
            else:
                assert line_list[4] == '#encode#'
                encode_tokens = line_list[-1].split(' ')
                encode_tokens = list(map(int,encode_tokens))
            padded_encode_tokens = pad_func(encode_tokens,50)

            if qid != tmp_qid:
                if len(tmp_ans):
                    tmp_ans, tmp_label = pad_none_sentence(tmp_ans, tmp_label, pos_num)
                    pos_answers.append(tmp_ans)
                    pos_label.append(tmp_label)
                tmp_ans = [padded_encode_tokens]
                if flag:
                    tmp_label = [0]
                else: 
                    tmp_label = [1]
                tmp_qid = qid
            else:
                if flag:
                    tmp_label.append(0)
                else: 
                    tmp_label.append(1)
                tmp_ans.append(padded_encode_tokens)
        tmp_ans,tmp_label = pad_none_sentence(tmp_ans, tmp_label, pos_num)
        pos_answers.append(tmp_ans)
        pos_label.append(tmp_label)
    pos_answers = np.array(pos_answers)
    pos_label = np.array(pos_label)
    neg_answers = pos_answers.copy()
    neg_answers = np.reshape(neg_answers,(-1,50))
    neg_answers = shuffle(neg_answers, random_state=17)
    neg_answers = np.reshape(neg_answers, (-1,3,50))
    assert pos_answers.shape == neg_answers.shape
    neg_label = np.zeros_like(pos_label)
    answers = np.concatenate([pos_answers,neg_answers],axis=1)
    labels = np.concatenate([pos_label, neg_label], axis=1)
    return answers, labels 

def rank_pos_bert_ans_ds(pos_file, pos_num):
    print('using rank_pos_bert_ans_ds func:  ')
    def pad_none_bert_sen(ans,label,num):
        assert len(ans) <= num, 'ans num greater than num: {}'.format(num)
        padding_sentence = [0. for _ in range(768)]
        for _ in range(num - len(ans)):
            ans.append(padding_sentence)
            label.append(0)
        return ans,label
    pos_answers = []
    pos_label = []
    with open(pos_file,'r',encoding='utf-8') as f1:
        tmp_ans = []
        tmp_label = []
        tmp_qid= ''
        for i, line in enumerate(f1):
            line_list = line.strip().split('\t')
            qid = line_list[0]
            
            assert line_list[-2] == '#EOS#'
            bert_vector = line_list[-1][1:-1]
            vector_list = bert_vector.split(', ')
            vector_list = list(map(float,vector_list))

            if qid != tmp_qid:
                if len(tmp_ans):
                    tmp_ans, tmp_label = pad_none_bert_sen(tmp_ans, tmp_label, pos_num)
                    pos_answers.append(tmp_ans)
                    pos_label.append(tmp_label)
                tmp_ans = [vector_list]
                tmp_label = [1]
                tmp_qid = qid
            else:
                tmp_label.append(1)
                tmp_ans.append(vector_list)
        tmp_ans,tmp_label = pad_none_bert_sen(tmp_ans, tmp_label, pos_num)
        pos_answers.append(tmp_ans)
        pos_label.append(tmp_label)
    pos_answers = np.array(pos_answers)
    pos_label = np.array(pos_label)
    neg_answers = pos_answers.copy()
    neg_answers = np.reshape(neg_answers,(-1,768))
    shuffle(neg_answers, random_state=17)
    neg_answers = np.reshape(neg_answers, (-1,pos_num,768))
    assert pos_answers.shape == neg_answers.shape
    neg_label = np.zeros_like(pos_label)
    answers = np.concatenate([pos_answers,neg_answers],axis=1)
    labels = np.concatenate([pos_label, neg_label], axis=1)
    return answers, labels 

def rank_ds(mode, batch_size, data_type,image_type,answer_file_index=1):
    print('calling rank_ds')
    if data_type == 2:
        questions, q_idxs = get_question_ds(file_path="data/type2_crop_only_questions/part_crop_only_questions_46k_encoded.txt")
        questions = np.array(questions)
        # first time filter corp words in sentence then cut word and encode: type2_ranked_3_pos_ans_encoded.txt
        # second time filter corp words in sentence then cut word and encode: type2_ranked_3_pos_ans_encoded_2.txt
        # remove corp word by index in encode list(line_list[5]), type2_ranked_3_pos_ans_2.txt
        # updata type2_ranked_3_pos_encoded.txt 

        # old MN-FT: CDR_CD_typex_image_vec_1280_xxxx.npy, new MN-FT: CDR_CD_MobileNet_typex_image_vec_1280_xxx.npy

        if answer_file_index == 1:
            answers, labels = rank_pos_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_2.txt",pos_num=3)
        elif answer_file_index == 2:
            answers, labels = rank_pos_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_encoded.txt",pos_num=3)
        elif answer_file_index == 3:
            del questions, q_idxs
            questions = np.load("data/type2_crop_only_questions/type2_question_46k_bert.npy")
            answers, labels = rank_pos_bert_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_bert.txt",pos_num=3)
        if image_type == 'MN':
            print('\n using new MN image vec! \n')
            #image_path = "data/type2_crop_only_questions/MN_type2_image_vec_1280d.npy"
            image_path = "data/type2_crop_only_questions/MN_type2_image_vec_1280_46252_new.npy"
        elif image_type == 'MN-FT':
            #image_path = "data/type2_crop_only_questions/CDR_CD_MobileNet_type2_image_vec_1280_46252.npy"
            image_path = "data/type2_crop_only_questions/CDR_CD_type2_image_vec_1280_46252.npy" 
        elif image_type == 'RN-FT':
            image_path = "data/type2_crop_only_questions/CDR_CD_ResNet_type2_image_vec_2048_46252.npy"
        elif image_type == 'XC-FT':
            image_path = "data/type2_crop_only_questions/CDR_CD_Xception_type2_image_vec_2048_46252.npy"
    
    elif data_type == 1:
        questions, q_idxs = get_question_ds(file_path="data/type1_useless_questions/part_useless_questions_2w_encoded.txt")
        questions = np.array(questions)
        if answer_file_index == 1:
            answers, labels = rank_pos_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_2.txt",pos_num=3)
        elif answer_file_index == 2:
            answers, labels = rank_pos_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_encoded.txt",pos_num=3)
        elif answer_file_index == 3:
            del questions, q_idxs
            questions = np.load("data/type1_useless_questions/type1_question_2w_bert.npy")
            answers, labels = rank_pos_bert_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_bert.txt",pos_num=3)
        if image_type == 'MN':
            print('\nusing new MN image vec! \n')
            image_path = "data/type1_useless_questions/MN_type1_image_vec_1280_20708_new.npy"
            #image_path = "data/type1_useless_questions/MN_type1_image_vec_1280d.npy"
        elif image_type == "MN-FT":
            #image_path = "tmp/tmp_CDR_CD_MobileNet_type1_image_vec_1280_20708.npy"
            #print("type1 data MN-FT file: tmp/tmp_CDR_CD_MobileNet_type1_image_vec_1280_20708.npy")
            #image_path = "data/type1_useless_questions/CDR_CD_MobileNet_type1_image_vec_1280_20708.npy"
            image_path = "data/type1_useless_questions/CDR_CD_type1_image_vec_1280_20708.npy"
        elif image_type == "RN-FT":
            image_path = "data/type1_useless_questions/CDR_CD_ResNet_type1_image_vec_2048_20708.npy"
        elif image_type == "XC-FT":
            image_path = "data/type1_useless_questions/CDR_CD_Xception_type1_image_vec_2048_20708.npy"
    
    elif data_type == 3:
        questions, q_idxs = get_question_ds(file_path="data/type3/type3_questions_encoded.txt")
        questions = np.array(questions)
        if answer_file_index == 1:
            answers, labels = rank_pos_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_2.txt",pos_num=3)
        elif answer_file_index == 2:
            answers, labels = rank_pos_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_encoded.txt",pos_num=3)
        elif answer_file_index == 3:
            del questions, q_idxs
            questions = np.load("data/type3/type3_question_2w_bert.npy")
            answers, labels = rank_pos_bert_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_bert.txt",pos_num=3)

        if image_type == 'MN':
            image_path = "data/type3/MN_type3_image_vec_1280_20295_new.npy"
        elif image_type == "MN-FT":
            #image_path = "data/type3/CDR_CD_MobileNet_type3_image_vec_1280_20295.npy"
            image_path = "data/type3/CDR_CD_type3_image_vec_1280_20295.npy"
        elif image_type == "RN-FT":
            image_path = "data/type3/CDR_CD_ResNet_type3_image_vec_2048_20295.npy"
        elif image_type == "XC-FT":
            image_path = "data/type3/CDR_CD_Xception_type3_image_vec_2048_20295.npy"
        
    images = np.load(image_path)
    assert questions.shape[0] == answers.shape[0] == images.shape[0] == labels.shape[0]
    questions = shuffle(questions, random_state=config.data_split_random_state)
    answers = shuffle(answers, random_state=config.data_split_random_state)
    images = shuffle(images, random_state=config.data_split_random_state)
    labels = shuffle(labels, random_state=config.data_split_random_state)
    if data_type == 2:
        questions = questions[:20295]
        answers = answers[:20295]
        images = images[:20295]
        labels = labels[:20295]
     
    total_num = questions.shape[0]
    print('total num {} of data type {} '.format(total_num,data_type))
    #test_q_idxs = q_idxs[int(0.95*total_num):]
    if mode == 'train':
        Q = questions[:int(0.85*total_num)]
        A = answers[:int(0.85*total_num)]
        V = images[:int(0.85*total_num)]
        Y = labels[:int(0.85*total_num)]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'val':
        Q = questions[int(0.85*total_num):int(0.95*total_num)]
        A = answers[int(0.85*total_num):int(0.95*total_num)]
        V = images[int(0.85*total_num):int(0.95*total_num)]
        Y = labels[int(0.85*total_num):int(0.95*total_num)]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'test':
        Q = questions[int(0.95*total_num):]
        A = answers[int(0.95*total_num):]
        V = images[int(0.95*total_num):]
        Y = labels[int(0.95*total_num):]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    else:
        raise KeyError('parameter "mode" must be train,val or test')
    feature_ds = tf.data.Dataset.from_tensor_slices((Q,A,V))
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(Y, tf.float32))
    ds = tf.data.Dataset.zip((feature_ds,label_ds))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache()
    if mode == 'train':
        #pass
        ds = ds.shuffle(buffer_size=int(total_num*0.85))
    if mode == 'val':
        #pase
        ds = ds.shuffle(buffer_size=int(total_num*0.1))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def rank_ds_mix(mode,batch_size,image_type):
    print('calling rank_ds_mix func')
    question_1, q_idx_1 = get_question_ds(file_path="data/type1_useless_questions/part_useless_questions_2w_encoded.txt")
    answer_1, label_1 = rank_pos_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_encoded.txt",pos_num=3)
    question_2, q_idx_2 = get_question_ds(file_path="data/type2_crop_only_questions/part_crop_only_questions_46k_encoded.txt")
    answer_2, label_2 = rank_pos_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_encoded.txt",pos_num=3)
    question_3, q_idx_3 = get_question_ds(file_path="data/type3/type3_questions_encoded.txt")
    answer_3, label_3 = rank_pos_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_encoded.txt",pos_num=3)
    if image_type == "MN":
        #print('\nusing old MN image vec! \n')
        #type1_image_path = "data/type1_useless_questions/MN_type1_image_vec_1280d.npy"
        #type2_image_path = "data/type2_crop_only_questions/MN_type2_image_vec_1280d.npy"
        #type3_image_path = "data/type3/MN_type3_image_vec_1280_20295_new.npy"
        print('\n using new MN image vec! \n')
        type1_image_path = r"data/type1_useless_questions/MN_type1_image_vec_1280_20708_new.npy"
        type2_image_path = r"data/type2_crop_only_questions/MN_type2_image_vec_1280_46252_new.npy"
        type3_image_path = r"data/type3/MN_type3_image_vec_1280_20295_new.npy"
    elif image_type == "MN-FT":
        type1_image_path = "data/type1_useless_questions/CDR_CD_type1_image_vec_1280_20708.npy"
        type2_image_path = "data/type2_crop_only_questions/CDR_CD_type2_image_vec_1280_46252.npy"
        type3_image_path = "data/type3/CDR_CD_type3_image_vec_1280_20295.npy"
    elif image_type == "RN-FT":
        type1_image_path = "data/type1_useless_questions/CDR_CD_ResNet_type1_image_vec_2048_20708.npy"
        type2_image_path = "data/type2_crop_only_questions/CDR_CD_ResNet_type2_image_vec_2048_46252.npy"
        type3_image_path = "data/type3/CDR_CD_ResNet_type3_image_vec_2048_20295.npy"
    elif image_type == 'XC-FT':
        type1_image_path = "data/type1_useless_questions/CDR_CD_Xception_type1_image_vec_2048_20708.npy"
        type2_image_path = "data/type2_crop_only_questions/CDR_CD_Xception_type2_image_vec_2048_46252.npy"
        type3_image_path = "data/type3/CDR_CD_Xception_type3_image_vec_2048_20295.npy"

    type1_images = np.load(type1_image_path)
    type2_images = np.load(type2_image_path)
    type3_images = np.load(type3_image_path)

    question_1.extend(question_2)
    question_1.extend(question_3)
    questions = np.array(question_1)
    print("all question shape: ",questions.shape)
    answers = np.concatenate([answer_1, answer_2, answer_3],axis=0)
    print("all answer shape: ",answers.shape) 
    all_image_vec = np.concatenate([type1_images, type2_images, type3_images],axis=0)
    print("all images vec shape: ",np.array(all_image_vec).shape)
    labels = np.concatenate([label_1,label_2,label_3],axis=0)
    print("all labels shape: ",labels.shape)

    questions = shuffle(questions, random_state=config.data_split_random_state)
    answers = shuffle(answers, random_state=config.data_split_random_state)
    images = shuffle(all_image_vec, random_state=config.data_split_random_state)
    labels = shuffle(labels, random_state=config.data_split_random_state)
    
    total_num = questions.shape[0]
    
    if mode == 'train':
        Q = questions[:int(0.85*total_num)]
        A = answers[:int(0.85*total_num)]
        V = images[:int(0.85*total_num)]
        Y = labels[:int(0.85*total_num)]
        assert Q.shape[0] == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'val':
        Q = questions[int(0.85*total_num):int(0.95*total_num)]
        A = answers[int(0.85*total_num):int(0.95*total_num)]
        V = images[int(0.85*total_num):int(0.95*total_num)]
        Y = labels[int(0.85*total_num):int(0.95*total_num)]
        assert Q.shape[0] == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'test':
        Q = questions[int(0.95*total_num):]
        A = answers[int(0.95*total_num):]
        V = images[int(0.95*total_num):]
        Y = labels[int(0.95*total_num):]
        assert Q.shape[0] == A.shape[0] == Y.shape[0] == V.shape[0]
    else:
        raise KeyError('parameter "mode" must be train,val or test')
    feature_ds = tf.data.Dataset.from_tensor_slices((Q,A,V))
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(Y, tf.float32))
    ds = tf.data.Dataset.zip((feature_ds,label_ds))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache()
    if mode == 'train':
        ds = ds.shuffle(buffer_size=int(total_num*0.85))
    if mode == 'val':
        ds = ds.shuffle(buffer_size=int(total_num*0.1))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds    

def rank_ds_all_img_vec(mode,batch_size,data_type,image_type):
    print('calling rank_ds_all_img_vec func, model: {}, batch_size: {}, data_type: {}, iamge_type: {}'.format(mode,batch_size,data_type,image_type))
    if data_type == 1:
        questions, q_idxs = get_question_ds(file_path="data/type1_useless_questions/part_useless_questions_2w_encoded.txt")
        answers, labels = rank_pos_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_encoded.txt",pos_num=3)
        print("asnwer file: data/type1_useless_questions/type1_ranked_3_pos_ans_encoded.txt") 
        if image_type == 'MN':
            image_path = "data/type1_useless_questions/MN_type1_image_vec_771280_20708.npy"
        elif image_type == "MN-FT":
            image_path = "data/type1_useless_questions/CDR_CD_type1_image_vec_771280_20708.npy"
        
        images = np.load(image_path)
        print(images.shape)
        
    elif data_type == 2:
        questions, q_idxs = get_question_ds(file_path="data/type2_crop_only_questions/part_crop_only_questions_46k_encoded.txt")
        answers, labels = rank_pos_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_encoded.txt",pos_num=3)
        print("answer file : data/type2_crop_only_questions/type2_ranked_3_pos_ans_encoded.txt")
        if image_type == 'MN':
            image_path = "data/type2_crop_only_questions/MN_type2_image_vec_771280_46252.npy"
        elif image_type == 'MN-FT':
            image_path = "data/type2_crop_only_questions/CDR_CD_type2_image_vec_771280_46252.npy"
       
        images = np.load(image_path)
        print(images.shape)
       
    elif data_type == 3:
        questions, q_idxs = get_question_ds(file_path="data/type3/type3_questions_encoded.txt")
        answers, labels = rank_pos_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_encoded.txt",pos_num=3)
        print("answer file: data/type3/type3_ranked_3_pos_ans_encoded.txt")
        if image_type == 'MN':
            image_path = "data/type3/MN_type3_image_vec_771280_20295.npy"
        elif image_type == "MN-FT":
            image_path = "data/type3/CDR_CD_type3_image_vec_771280_20295.npy"
        images = np.load(image_path)
        print(images.shape)
        
    elif data_type == 6:
        question_1, q_idx_1 = get_question_ds(file_path="data/type1_useless_questions/part_useless_questions_2w_encoded.txt")
        answer_1, label_1 = rank_pos_ans_ds(pos_file="data/type1_useless_questions/type1_ranked_3_pos_ans_encoded.txt",pos_num=3)
        question_2, q_idx_2 = get_question_ds(file_path="data/type2_crop_only_questions/part_crop_only_questions_46k_encoded.txt")
        answer_2, label_2 = rank_pos_ans_ds(pos_file="data/type2_crop_only_questions/type2_ranked_3_pos_ans_encoded.txt",pos_num=3)
        question_3, q_idx_3 = get_question_ds(file_path="data/type3/type3_questions_encoded.txt")
        answer_3, label_3 = rank_pos_ans_ds(pos_file="data/type3/type3_ranked_3_pos_ans_encoded.txt",pos_num=3)
        if image_type == "MN":
            print('\n using new MN image vec! \n')
            type1_image_path = "data/type1_useless_questions/MN_type1_image_vec_771280_20708.npy"
            type2_image_path = "data/type2_crop_only_questions/MN_type2_image_vec_771280_46252.npy"
            type3_image_path = "data/type3/MN_type3_image_vec_771280_20295.npy"
        elif image_type == "MN-FT":
            type1_image_path = "data/type1_useless_questions/CDR_CD_type1_image_vec_771280_20708.npy"
            type2_image_path = "data/type2_crop_only_questions/CDR_CD_type2_image_vec_771280_46252.npy"
            type3_image_path = "data/type3/CDR_CD_type3_image_vec_771280_20295.npy"
        type1_images = np.load(type1_image_path)
        type2_images = np.load(type2_image_path)
        type3_images = np.load(type3_image_path)

        question_1.extend(question_2)
        question_1.extend(question_3)
        questions = np.array(question_1)
        print("all question shape: ",questions.shape)
        answers = np.concatenate([answer_1, answer_2, answer_3],axis=0)
        print("all answer shape: ",answers.shape)
        images = np.concatenate([type1_images, type2_images, type3_images],axis=0)
        print("all images vec shape: ",images.shape)
        labels = np.concatenate([label_1,label_2,label_3],axis=0)
        print("all labels shape: ",labels.shape)
   
    else:
        KeyError('param data_type: {} is not included'.format(data_type))

    questions = shuffle(questions, random_state=config.data_split_random_state)
    answers = shuffle(answers, random_state=config.data_split_random_state)
    images = shuffle(images, random_state=config.data_split_random_state)
    labels = shuffle(labels, random_state=config.data_split_random_state)
    if data_type == 2:
        questions = questions[:20295]
        answers = answers[:20295]
        images = images[:20295]
        labels = labels[:20295]
   
    total_num = len(questions)
    print('total num {} of data type {} '.format(total_num,data_type))
    #test_q_idxs = q_idxs[int(0.95*total_num):]
    if mode == 'train':
        Q = questions[:int(0.85*total_num)]
        A = answers[:int(0.85*total_num)]
        V = images[:int(0.85*total_num)]
        Y = labels[:int(0.85*total_num)]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'val':
        Q = questions[int(0.85*total_num):int(0.95*total_num)]
        A = answers[int(0.85*total_num):int(0.95*total_num)]
        V = images[int(0.85*total_num):int(0.95*total_num)]
        Y = labels[int(0.85*total_num):int(0.95*total_num)]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    elif mode == 'test':
        Q = questions[int(0.95*total_num):]
        A = answers[int(0.95*total_num):]
        V = images[int(0.95*total_num):]
        Y = labels[int(0.95*total_num):]
        assert len(Q) == A.shape[0] == Y.shape[0] == V.shape[0]
    else:
        raise KeyError('parameter "mode" must be train,val or test')
    feature_ds = tf.data.Dataset.from_tensor_slices((Q,A,V))
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(Y, tf.float32))
    ds = tf.data.Dataset.zip((feature_ds,label_ds))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache()
    if mode == 'train':
        ds = ds.shuffle(buffer_size=int(total_num*0.85))
    if mode == 'val':
        ds = ds.shuffle(buffer_size=int(total_num*0.1))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

    
        

