import os
import tensorflow.compat.v1 as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)

model_dir = os.path.join(file_path, 'chinese_L-12_H-768_A-12/')
config_name = os.path.join(model_dir, 'bert_config.json')
ckpt_name = os.path.join(model_dir, 'bert_model.ckpt')
output_dir = os.path.join(file_path, 'tmp/result/')
vocab_file = os.path.join(model_dir, 'vocab.txt')
data_dir = os.path.join(file_path, 'data/')

# print(output_dir, file_path)

num_train_epochs = 20
batch_size = 128
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_len = 200

# graph名字
graph_name = 'graph'
graph_file = os.path.join(output_dir, graph_name)