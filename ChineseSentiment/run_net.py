
import os
import sys
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torchviz import make_dot
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from .load_data import id2vocab_file, train_id_file, train_lable_file, valid_id_file, valid_lable_file, test_id_file, test_lable_file
from .mlp import load_mlp
from .cnn import load_cnn
from .rnn import load_rnn
from .rcnn import load_rcnn
from .cnn_ori import load_cnn_origin


def load_files():
    global train_text, train_lable, valid_text, valid_lable, test_text, test_lable
    train_text = np.load(train_id_file, allow_pickle=True)
    train_lable = np.load(train_lable_file, allow_pickle=True)
    valid_text = np.load(valid_id_file, allow_pickle=True)
    valid_lable = np.load(valid_lable_file, allow_pickle=True)
    test_text = np.load(test_id_file, allow_pickle=True)
    test_lable = np.load(test_lable_file, allow_pickle=True)


def batch_generator(text, lable, batch_size, training=True):
    batch_num = len(text) // batch_size
    for batch in range(batch_num):
        input, target, lens = [], [], []
        if training:
            idx = torch.randperm(batch_size)
        else:
            idx = torch.arange(batch_size)
        for sen in idx:
            temp_text = text[batch*batch_size+sen]
            lens.append(len(temp_text))
            input.append(torch.Tensor(temp_text))
            target.append(lable[batch*batch_size+sen])
        input = pad_sequence(input, batch_first=True).long().to(device)
        target = torch.Tensor(target).long().to(device)
        lens = torch.Tensor(lens).long().to(device)
        yield input, target, lens


def evaluate(output, target, average='macro'):  # 'macro' or 'micro'
    pred = torch.argmax(output, dim=1).cpu()
    groud_truth = torch.argmax(target, dim=1).cpu()
    # print(pred.shape, groud_truth.shape)
    acc = accuracy_score(pred, groud_truth)
    f1 = f1_score(pred, groud_truth, average=average)
    co = pearsonr(pred, groud_truth)[0]
    if np.isnan(co):
        co = 0.0
    return acc, f1, co


def train_func(model, optimizer, batch_size):
    model.train()
    pred_list, target_list = [], []
    train_loss = []
    for input, target, _ in batch_generator(train_text, train_lable, batch_size):
        optimizer.zero_grad()
        loss, pred = model(input, target, dropout=True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        pred_list.append(pred)
        target_list.append(target)
    acc, f1, co = evaluate(torch.cat(pred_list, 0), torch.cat(target_list, 0))
    return np.mean(train_loss), acc, f1, co, input, target


def valid_func(model, batch_size):
    model.eval()
    pred_list, target_list = [], []
    with torch.no_grad():
        valid_loss = []
        for input, target, _ in batch_generator(valid_text, valid_lable, batch_size, training=False):
            loss, pred = model(input, target, dropout=False)
            valid_loss.append(loss.item())
            pred_list.append(pred)
            target_list.append(target)
    acc, f1, co = evaluate(torch.cat(pred_list, 0), torch.cat(target_list, 0))
    return np.mean(valid_loss), acc, f1, co


def test_func(model, batch_size):
    model.eval()
    pred_list, target_list = [], []
    with torch.no_grad():
        test_loss = []
        for input, target, _ in batch_generator(test_text, test_lable, batch_size, training=False):
            loss, pred = model(input, target, dropout=False)
            test_loss.append(loss.item())
            pred_list.append(pred)
            target_list.append(target)
    acc, f1, co = evaluate(torch.cat(pred_list, 0), torch.cat(target_list, 0))
    return np.mean(test_loss), acc, f1, co


def save_config(config):
    print(config)
    os.makedirs(config['output_path'], exist_ok=True)
    with open(os.path.join(config['output_path'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=1)


def load_model(config, net_name):
    if config['net'] == 'mlp':
        model = load_mlp(config, os.path.join(config['output_path'], net_name + '.pkl'), device)
    elif config['net'] == 'textcnn':
        model = load_cnn(config, os.path.join(config['output_path'], net_name + '.pkl'), device)
    elif config['net'] == 'rnn':
        model = load_rnn(config, os.path.join(config['output_path'], net_name + '.pkl'), device)
    elif config['net'] == 'cnn':
        model = load_cnn_origin(config, os.path.join(config['output_path'], net_name + '.pkl'), device)
    elif config['net'] == 'rcnn':
        model = load_rcnn(config, os.path.join(config['output_path'], net_name + '.pkl'), device)
    return model


def writer_add_scalar(writer, loss, acc, f1, corr, epoch, mode='train'):
    if mode == 'train':
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar('train_F1_macro', f1, epoch)
        writer.add_scalar('train_CORR', corr, epoch)
    else:
        writer.add_scalar('dev_loss', loss, epoch)
        writer.add_scalar('dev_acc', acc, epoch)
        writer.add_scalar('dev_F1_macro', f1, epoch)
        writer.add_scalar('dev_CORR', corr, epoch)


def run_net(config):
    global device
    device = config['device']
    os.makedirs(config['output_path'], exist_ok=True)
    save_config(config)
    if config['net'] == 'rnn' or config['net'] == 'rcnn':
        net_name = config['net'] + "_" + config['rnn']
    else:
        net_name = config['net']
    print("Running Model: " + net_name + "...")
    if config['output_path'] != 'save/all':
        writer = SummaryWriter(log_dir=os.path.join(config['output_path'], 'run'))
    else:
        writer = SummaryWriter(log_dir=os.path.join(config['output_path'], 'run/' + net_name))    
    # load model  and  initial optimizer
    model = load_model(config, net_name)
    optimizer = optim.Adagrad(model.parameters(), lr=config['lr'], lr_decay=config['lr_decay'], weight_decay=config['weight_decay'])
    batch_size = config['batch_size']
    best_acc = 0.0
    best_epoch = 0
    begin_time = time.time()
    # 重定向
    __console__ = sys.stderr
    sys.stderr = open(os.path.join(config['output_path'], net_name) + '.log', 'w')
    # train and valid
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_f1, train_co, inputs, targets = train_func(model, optimizer, batch_size)
        valid_loss, valid_acc, valid_f1, valid_co = valid_func(model, batch_size)
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
        sys.stderr.write(f'epoch:{epoch}\n\tbest_epoch:{best_epoch}\tbest_acc:{best_acc}\n')
        sys.stderr.write(f'\ttrain loss:{train_loss}\ttrain acc:{train_acc}\ttrain F1:{train_f1}\ttrain CORR:{train_co}\n')
        sys.stderr.write(f'\tvalid loss:{valid_loss}\tvalid acc:{valid_acc}\tvalid F1:{valid_f1}\tvalid CORR:{valid_co}\n')
        print(f"| epoch:{epoch:3d}| train acc: {train_acc:.5f}| valid acc: {valid_acc:.5f}| best_acc: {best_acc:.5f}| best_epoch: {best_epoch:3d}|")
        # tensor board
        writer_add_scalar(writer, train_loss, train_acc, train_f1, train_co, epoch, mode='train')
        writer_add_scalar(writer, valid_loss, valid_acc, valid_f1, valid_co, epoch, mode='valid')
    end_time = time.time()
    # test
    test_loss, test_acc, test_f1, test_co = test_func(model, batch_size)
    sys.stderr.write(f'test loss:{test_loss}\ttest acc:{test_acc}\ttest F1:{test_f1}\ttest CORR:{test_co}\n')
    sys.stderr.write("Total Time Use: %d s\n" % (end_time - begin_time))
    sys.stderr = __console__
    # save model
    torch.save(model.state_dict(), os.path.join(config['output_path'], net_name + '.pkl'))
    # plot graph
    aloss, alables = model(inputs, targets)
    g = make_dot((aloss, alables), params=dict(model.named_parameters()))
    g.render(os.path.join(config['output_path'], net_name), view=False)
    writer.close()