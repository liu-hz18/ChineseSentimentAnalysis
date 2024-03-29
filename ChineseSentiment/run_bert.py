
import sys
import numpy as np
import random
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score

from .bert.extract_feature import BertVector
from .run_net import writer_add_scalar
from .FocalLoss import FocalLoss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_file = 'sina/sinanews.train'
valid_file = 'sina/sinanews.valid'
test_file = 'sina/sinanews.test'


class TextSentimentBERT(nn.Module):
    def __init__(self, embed_dim=768, mlp_hidden_size=512, num_layers=1, batch_size=32,
                 drop_out=0.2, num_class=8, device='cuda:0', loss_type='cross_entropy'):
        super(TextSentimentBERT, self).__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.mlp_hidden_size2 = 64
        self.device = torch.device(device)
        self.drop_out = nn.Dropout(drop_out)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(mlp_hidden_size, self.mlp_hidden_size2),
            nn.BatchNorm1d(self.mlp_hidden_size2),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(self.mlp_hidden_size2, num_class)
        )
        self.init_loss(loss_type)
        print('init finished')

    def forward(self, embed, target, dropout=True):
        if dropout:
            embed = self.drop_out(embed)
        output = self.mlp(embed)
        labels = F.softmax(output, dim=1)
        if self.loss_type == 'cross_entropy' or self.loss_type == 'focal':
            loss = self.loss_fn(output, torch.argmax(target, dim=1))
        else:
            loss = self.loss_fn(labels, target.float())
        return loss, labels.detach()

    def init_loss(self, loss_type):
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(self.num_class)
        else:
            self.loss_type = 'cross_entropy'
            self.loss_fn = nn.CrossEntropyLoss()


def sentence_process(line):
    blocks = line.strip().split('\t')
    sentiments, sentences = blocks[1].strip().split(), blocks[2].strip().split()
    lable = [int(score.strip().split(':')[-1]) for score in sentiments[1:]]
    return lable, "".join(sentences)


def init_data(file_name):
    print('init data:' + file_name)
    lable_list = []
    sentence_list = []
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            lable, sentence = sentence_process(line)
            lable_list.append(lable)
            sentence_list.append(bert.encode([sentence]))
    return np.array(lable_list), np.array(sentence_list)


def batch_generator(lable_list, sentence_list, batch_size, training=True):
    batch_num = len(lable_list) // batch_size
    for batch in range(batch_num):
        input, target = sentence_list[batch*batch_size:(batch+1)*batch_size], lable_list[batch*batch_size:(batch+1)*batch_size]
        if training:
            idx = torch.randperm(batch_size)
            real_input = np.zeros((input.shape[0], input.shape[1]))
            real_lable = np.zeros((target.shape[0], target.shape[1]))
            for rank, sen in enumerate(idx):
                real_input[rank] = input[sen]
                real_lable[rank] = target[sen]
            input = real_input
            target = real_lable
        yield torch.Tensor(input).to(device), torch.Tensor(target).to(device)


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
    for input, target in batch_generator(train_lable, train_text, batch_size):
        optimizer.zero_grad()
        loss, pred = model(input, target, dropout=True)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        pred_list.append(pred)
        target_list.append(target)
    acc, f1, co = evaluate(torch.cat(pred_list, 0), torch.cat(target_list, 0))
    return np.mean(train_loss), acc, f1, co


def valid_func(model, batch_size):
    model.eval()
    pred_list, target_list = [], []
    with torch.no_grad():
        valid_loss = []
        for input, target in batch_generator(valid_lable, valid_text, batch_size, training=False):
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
        for input, target in batch_generator(test_lable, test_text, batch_size, training=False):
            loss, pred = model(input, target, dropout=False)
            test_loss.append(loss.item())
            pred_list.append(pred)
            target_list.append(target)
    acc, f1, co = evaluate(torch.cat(pred_list, 0), torch.cat(target_list, 0))
    return np.mean(test_loss), acc, f1, co


def run_net(config):
    model = TextSentimentBERT().to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=config['lr'], lr_decay=config['lr_decay'], weight_decay=config['weight_decay'])
    best_acc = 0.0
    best_epoch = 0
    batch_size = config['batch_size']
    writer = SummaryWriter(log_dir=os.path.join(config['output_path'], 'run'))
    # train and valid
    print('run epochs')
    begin_time = time.time()
    for epoch in range(config['epochs']):
        train_loss, train_acc, train_f1, train_co = train_func(model, optimizer, batch_size)
        valid_loss, valid_acc, valid_f1, valid_co = valid_func(model, batch_size)
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch
            test_loss, test_acc, test_f1, test_co = test_func(model, batch_size)
        sys.stdout.write(f'epoch:{epoch}\n\tbest_epoch:{best_epoch}\tbest_acc:{best_acc}\ttest_acc:{test_acc}\ttest F1:{test_f1}\ttest CORR:{test_co}\n')
        sys.stdout.write(f'\ttrain loss:{train_loss}\ttrain acc:{train_acc}\ttrain F1:{train_f1}\ttrain CORR:{train_co}\n')
        sys.stdout.write(f'\tvalid loss:{valid_loss}\tvalid acc:{valid_acc}\tvalid F1:{valid_f1}\tvalid CORR:{valid_co}\n')
        # tensor board
        writer_add_scalar(writer, train_loss, train_acc, train_f1, train_co, epoch, mode='train')
        writer_add_scalar(writer, valid_loss, valid_acc, valid_f1, valid_co, epoch, mode='valid')
    end_time = time.time()
    print("Time use:", end_time - begin_time)
    writer.close()


def preprocess_bert():
    global bert
    bert = BertVector()
    train_lable, train_text = init_data(train_file)
    valid_lable, valid_text = init_data(valid_file)
    test_lable, test_text = init_data(test_file)
    np.save('bert-pretrained/train_tensor', train_text.reshape(train_text.shape[0], -1))
    np.save('bert-pretrained/train_lable', train_lable)
    np.save('bert-pretrained/valid_tensor', valid_text.reshape(valid_text.shape[0], -1))
    np.save('bert-pretrained/valid_lable', valid_lable)
    np.save('bert-pretrained/test_tensor', test_text.reshape(test_text.shape[0], -1))
    np.save('bert-pretrained/test_lable', test_lable)
    print(test_text)
    print(test_text.shape)


def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


config = {
    'batch_size': 64,
    'epochs': 400,
    'lr': 0.01,
    'lr_decay': 0.0,
    'weight_decay': 0.0001,
    'seed': 11446,
    'output_path': 'save/bert',
    # 'output_path': 'save/bert_dp',
    # 'output_path': 'save/bert_bn',
    # 'output_path': 'save/bert_no',
}


def run_bert():
    global train_text, valid_text, test_text, train_lable, valid_lable, test_lable
    os.makedirs(config['output_path'], exist_ok=True)
    train_text = np.load('bert-pretrained/train_tensor.npy')
    valid_text = np.load('bert-pretrained/valid_tensor.npy')
    test_text = np.load('bert-pretrained/test_tensor.npy')
    train_lable = np.load('bert-pretrained/train_lable.npy')
    valid_lable = np.load('bert-pretrained/valid_lable.npy')
    test_lable = np.load('bert-pretrained/test_lable.npy')
    run_net(config)


if __name__ == '__main__':
    init_seed(config['seed'])
    # preprocess_bert()
    run_bert()
