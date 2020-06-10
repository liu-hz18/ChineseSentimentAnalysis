
import torch
import random
import sys
import time
from ChineseSentiment.load_data import preprocess
from ChineseSentiment.run_net import run_net, load_files
from ChineseSentiment.run_bert import run_bert

VOCAB_SIZE = 45818
NUM_CLASS = 8
cool_down_time = 3
# To Visualization:
# ->  tensorboard --logdir save/all/run

basic_config = {
    'use_pretrained': False,
    'vocab_size': VOCAB_SIZE,
    'embed_size': 300,
    'num_class': NUM_CLASS,
    'seed': 11446,
    'embed_path': 'word2vec/matrix.npy',
    'device': "cuda:0" if torch.cuda.is_available() else "cpu",
}


def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_mlp(path='mlp'):
    config = {
        'net': 'mlp',
        'output_path': 'save/' + path,  # 'save/mlp',
        'hidden_sizes': [256, 64],
        'max_length': 128,
        'batch_size': 64,
        'num_layers': 2,
        'drop_out': 0.6,
        'lr': 0.001,
        'epochs': 400,  # 100,
        'lr_decay': 0.0,
        'weight_decay': 0.0001,
        'loss_type': 'cross_entropy'  # 'focal'
    }
    run_net(dict(basic_config, **config))
    time.sleep(cool_down_time)


def run_cnn(path='cnn'):  # 57.6
    config = {
        'net': 'cnn',
        'output_path': 'save/' + path,  # 'save/cnn',
        'max_length': 256,
        'kernel_size': 3,
        'padding': 2,
        'conv_stride': 1,
        'out_channel': 512,  # 256,
        'pool_kernel_size': 2,  # 4,
        'pool_stride': 2,  # 4,
        'batch_size': 64,
        'drop_out': 0.6,  # 0.5,
        'mlp_hidden_size': 512,
        'lr': 0.001,
        'epochs': 400,  # 200,
        'lr_decay': 0.0,
        'weight_decay': 0.001,
        'loss_type': 'cross_entropy'  # 'focal'
    }
    run_net(dict(basic_config, **config))
    time.sleep(cool_down_time)


def run_textcnn(path='textcnn'):  # best: 61.3
    config = {
        'net': 'textcnn',
        'output_path': 'save/' + path,  # 'save/textcnn',
        'kernel_sizes': [2, 3, 4, 5],  # multiple region size
        'kernel_num': 100,  # feature maps for each region size
        'batch_size': 64,
        'drop_out': 0.5,
        'mlp_hidden_size': 512,
        'lr': 0.02,
        'epochs': 500,  # 200,
        'lr_decay': 0.0,
        'weight_decay': 0.001,
        'loss_type': 'cross_entropy'  # 'focal'
    }
    run_net(dict(basic_config, **config))
    time.sleep(cool_down_time)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_rnn(rnn_type='gru', path='rnn'):
    config = {
        'net': 'rnn',
        'output_path': 'save/' + path,  # 'save/rnn',
        'rnn': rnn_type,  # 'rnn' or 'lstm' or 'gru'
        'batch_size': 64,
        'drop_out': 0.5,
        'hidden_size': 128,
        'num_layers': 1,  # suggested 1 for bidirectional
        'bidirectional': True,

        'attention': True,
        'param_da': 350,  # for Attention
        'param_r': 30,  # for Attention

        'lr': 0.01,
        'epochs': 400,  # 150,
        'lr_decay': 0.0,
        'weight_decay': 0.001,
        'mlp_hidden_size': 256,
        'loss_type': 'cross_entropy',  # 'focal'
    }
    run_net(dict(basic_config, **config))
    time.sleep(cool_down_time)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_rcnn(rnn_type='gru', path='rcnn'):  # best: 60.9
    config = {
        'net': 'rcnn',
        'output_path': 'save/' + path,  # 'save/rcnn',
        'rnn': rnn_type,  # 'rnn' or 'lstm' or 'gru'
        'batch_size': 64,
        'drop_out': 0.5,
        'rnn_hidden_size': 256,
        'num_layers': 1,  # suggested 1 for bidirectional
        'mlp_1_hidden_size': 512,
        'mlp_2_hidden_size': 128,
        'lr': 0.01,
        'epochs': 400,  # 150,
        'lr_decay': 0.0,
        'weight_decay': 0.001,
        'loss_type': 'cross_entropy'  # 'focal'
    }
    run_net(dict(basic_config, **config))
    time.sleep(cool_down_time)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all():
    # MLP
    run_mlp(path='all')
    run_bert(path='all')
    # CNN
    run_cnn(path='all')
    run_textcnn(path='all')
    # RNN
    run_rnn(rnn_type='rnn', path='all')
    run_rnn(rnn_type='lstm', path='all')
    run_rnn(rnn_type='gru', path='all')
    # RCNN
    run_rcnn(rnn_type='rnn', path='all')
    run_rcnn(rnn_type='lstm', path='all')
    run_rcnn(rnn_type='gru', path='all')


def usage():
    print('Usage: python main.py [mlp|cnn|rnn|textcnn|rcnn|bert]')
    print(' For Preprocess:')
    print('       python main.py pre')


if __name__ == '__main__':
    print('vocab_size:', VOCAB_SIZE)
    init_seed(basic_config['seed'])
    func_map = {
        'mlp': run_mlp, 'cnn': run_cnn, 'rnn': run_rnn, 'textcnn': run_textcnn,
        'rcnn': run_rcnn, 'bert': run_bert, 'all': run_all, 'pre': preprocess,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in func_map:
        usage()
    elif sys.argv[1] == 'pre':
        preprocess()
    else:
        load_files()
        func_map[sys.argv[1]]()
