# use this seed to shuffle data for split train/val/test
data_split_random_state = 70

# image model param
data_argu = True
image_scale = [-1,1] # [-1,1] or [0,1]
data_argu_seed = 1234
base_model = 'mobilenet' # 'mobilenet','resnet','efficientnet'
image_task = 'CD' # 'C', 'D', 'CD'

####################################
save_model_arch = False

embedding_dim = 200
q_length = 30
a_length = 50
reply_num = 6
q_feature = 64 # q dim after lstm encoding
a_feature = 64 # a dim after lstm encoding
v_feature =1280 # v dim after pre-trained image model
v_mid_feature = 64

epoch = 20
batch_size = 64
data_type = 6 #1,2,3, 6 means  mix data
image_type = 'MN-FT' # 'MN','MN-FT','RN-FT','XC-FT'
fusion = 'concat' # 'concat','bilinear','conv_fusion','mfb'
att_type = None  # v_guided,q_guided,co-att-p,co-att-a
att_id = 0 # different attention score 
learning_rate = 0.001

answer_file_index = 2 # 1 means typex_3_pos_ans_2.txt file, 2 means typex_3_pos_ans_encoded.txt, 3 means use bert vector
only_embed = False
lstm_initializer = 'glorot_normal'
lstm_init_scale = 1.0

#############
# with_v and with_q must have at least 1 is True
with_q = True
with_v = True
###############
# project q
fc1_initializer = 'glorot_normal'
fc1_init_scale = 0.5
#project v
fc2_initializer = 'glorot_normal'
fc2_init_scale = 1.0
# project a
fc3_initializer = fc1_initializer
fc3_init_scale = fc1_init_scale

#ranker
fc4_initializer = 'glorot_normal'
fc4_init_scale = 1.0
fc5_initializer = 'glorot_normal'
fc5_init_scale = 1.0



