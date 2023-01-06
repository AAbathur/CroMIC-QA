# image dataset for fine-tuning
import tensorflow as tf
import os
from sklearn.utils import shuffle
import utils
import data
import config
import numpy as np

def image_process(image_path):
    IMG_SIZE = 224
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image,tf.float32)
    image /= 255.
    return image

def image_process_2(image_path):
    IMG_SIZE = 224
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image,tf.float32)
    image /= 127.5
    image -= 1.0
    return image

def get_image_ds(mode,batch_size,CD,base_model):
    image_folder = r"image-data/images-134k"
    label_path = r"data/fine_tune_labels.txt"
    crop2idx = utils.get_vocab_idx_dict('data/crops.txt')
    dise2idx = utils.get_vocab_idx_dict("data/disease.txt")
    image_path = []
    crop_label = []
    dise_label = []
    with open(label_path,'r',encoding='utf-8') as f1:
        for line in f1:
            label_1 = [0. for _ in range(len(crop2idx))]
            label_2 = [0. for _ in range(len(dise2idx))]
            line_list = line.strip().split('\t')
            idx = line_list[0]
            crop = line_list[1]
            dise = line_list[2]
            if dise == 'None':
                continue
            image_path.append(os.path.join(image_folder,idx))
            label_1[crop2idx[crop]] = 1.
            label_2[dise2idx[dise]] = 1.
            crop_label.append(label_1)
            dise_label.append(label_2)
    seed = 75
    image_path = shuffle(image_path,random_state=seed)
    crop_label = shuffle(crop_label,random_state=seed)
    dise_label = shuffle(dise_label,random_state=seed)
    total_num = len(image_path)
    print('total num: ',total_num)
    if mode == 'train':
        v = image_path[:int(total_num * 0.85)]
        y1 = crop_label[:int(total_num * 0.85)]
        y2 = dise_label[:int(total_num * 0.85)]
        print('train num: ',len(v))

    elif mode == 'val':
        v = image_path[int(total_num * 0.85):int(total_num * 0.95)]
        y1 = crop_label[int(total_num * 0.85):int(total_num * 0.95)]
        y2 = dise_label[int(total_num * 0.85):int(total_num * 0.95)]
        print('val num: ',len(v))
    elif mode == 'test':
        v = image_path[int(total_num * 0.95):]
        y1 = crop_label[int(total_num * 0.95):]
        y2 = dise_label[int(total_num * 0.95):]
        print('test num: ',len(v))
    else:
        raise KeyError('wrong mode parameter')
    v_ds = tf.data.Dataset.from_tensor_slices(v)
    if CD == 'C':
        y_ds = tf.data.Dataset.from_tensor_slices(y1)
    elif CD == 'D':
        y_ds = tf.data.Dataset.from_tensor_slices(y2)
    elif CD == 'CD':
        y_ds = tf.data.Dataset.from_tensor_slices((y1,y2))
    else:
        raise KeyError('parameter CD wrong')
    if base_model == 'mobilenet':
         v_ds = v_ds.map(image_process)
    elif base_model == 'resnet' or base_model == 'xception':
         v_ds = v_ds.map(image_process_2)
    ds = tf.data.Dataset.zip((v_ds, y_ds))
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    ds = ds.cache()
    if mode == 'train':
        ds = ds.shuffle(buffer_size=int(total_num*0.85))
    if mode == 'val':
        #ds = ds.shuffle(buffer_size=int(total_num*0.1))
        pass
    
    ds = ds.batch(batch_size)
    #ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

def get_type1_image_ds(batch_size):
    _, q_idxs = data.get_question_ds(file_path="data/type1_useless_questions/part_useless_questions_2w_encoded.txt")
    type1_images = []
    image_folder = r"image-data/type1_images"
    for i, idx in enumerate(q_idxs):
        
        image_name = idx+".jpg"
        image_path = os.path.join(image_folder,image_name)
        type1_images.append(image_path)
    print('image num of type1: ', len(type1_images))
    v_ds = tf.data.Dataset.from_tensor_slices(type1_images)
    v_ds = v_ds.map(image_process_2)
    v_ds = v_ds.batch(batch_size)
    return v_ds

def get_type2_image_ds(batch_size):
    _, q_idxs = data.get_question_ds(file_path="data/type2_crop_only_questions/part_crop_only_questions_46k_encoded.txt")
    
    type2_images = []
    image_folder = r"image-data/type2_images"
    for i, idx in enumerate(q_idxs):
        if i< 40000: continue
        if i>=50000: 
             print('cut type2 top i: ',i)
             break
        image_name = idx+".jpg"
        image_path = os.path.join(image_folder,image_name)
        type2_images.append(image_path)
    print('image num of this time: ',len(type2_images))
    v_ds = tf.data.Dataset.from_tensor_slices(type2_images)
    v_ds = v_ds.map(image_process_2)
    v_ds = v_ds.batch(batch_size)
    return v_ds

def get_type3_image_ds(batch_size):
    _, q_idxs = data.get_question_ds(file_path="data/type3/type3_questions_encoded.txt")
    type3_images = []
    image_folder = r"image-data/type3_images"
    for i, idx in enumerate(q_idxs):
        image_name = idx+".jpg"
        image_path = os.path.join(image_folder,image_name)
        type3_images.append(image_path)
    print('image num of type3: ', len(type3_images))
    v_ds = tf.data.Dataset.from_tensor_slices(type3_images)
    v_ds = v_ds.map(image_process_2)
    v_ds = v_ds.batch(batch_size)
    return v_ds



    
        
