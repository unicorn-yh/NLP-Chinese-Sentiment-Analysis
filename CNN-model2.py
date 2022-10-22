import collections
from ctypes import sizeof
import os
import random
import time
import jieba
import pandas as pd
from snownlp import SnowNLP

import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchtext.legacy import vocab
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_size = 200

stopword_ls = []
def getStopWord():
    with open('lib/stopwords_utf8.txt', 'r',encoding='UTF-8') as file:
        for line in file:
            stopword_ls.append(line.split('\n')[0])

def isStopWord(word):
    for i in range(len(stopword_ls)):
        if word == stopword_ls[i]:
            return True
    return False

def concate_files(file1,file2,final_file):
      data = data2 = ""
      with open(file1, 'r', encoding='utf-8') as fp:  # Reading data from file1
            sentences = fp.readlines()
            for sentence in sentences[:int(dataset_size/2)]:
                  data += sentence
      with open(file2, 'r', encoding='utf-8') as fp:  # Reading data from file2
            sentences = fp.readlines()
            for sentence in sentences[:int(dataset_size/2)]:
                  data2 += sentence
      data += data2  # To add the data of file2
      with open (final_file, 'w', encoding='utf-8') as fp:
            fp.write(data)

concate_files('data/total/pos.txt','data/total/neg.txt','data/total/labeled_text.txt')
data = pd.read_csv('data/total/labeled_text.txt', header = None, delimiter='    ',encoding='utf-8',names=['Label', 'Text'])

#Assign Label to each sentence
PosNegLabel = []
for label in data.Label:
    if label == -1:        #negative
        PosNegLabel.append(0)
    elif label == 1:      #positive
        PosNegLabel.append(1)
data['Label']= PosNegLabel
data.head()

def snow_segment(text):
    word_ls = []
    tmp_ls = SnowNLP(text).words  #segmentation  
    for i in range(len(tmp_ls)):
        if not isStopWord(tmp_ls[i]):
            word_ls.append(tmp_ls[i]) 
    return word_ls

random.shuffle(data['Text'])
data['Segmented_Text'] = data['Text'].apply(lambda x: snow_segment(x))

data_train, data_test = train_test_split(data, test_size=0.20)

all_training_words = [word for tokens in data_train["Segmented_Text"] for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in data_train["Segmented_Text"]]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("TRAIN DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}".format(len(all_training_words),len(TRAINING_VOCAB),max(training_sentence_lengths)))

all_test_words = [word for tokens in data_test["Segmented_Text"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in data_test["Segmented_Text"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("TEST  DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}" .format(len(all_test_words),len(TEST_VOCAB),max(test_sentence_lengths)))

def get_vocab(data):
    tokenized_data = [words for words, _ in data]
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return vocab.Vocab(counter, min_freq=5)


def preprocess(data, vocab):  #data:list | vocab:torchtext.legacy.vocab.Vocab
    max_l = 500  # 将每条评论通过截断或者补0，使得长度变成500

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = [words for words, _ in data]
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


def load_pretrained_embedding(words, pretrained_vocab):
    """从预训练好的vocab中提取出words对应的词向量"""
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("Words Out of Vocabulary: {}".format(oov_count))
    return embed


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    opt_test_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in tqdm(train_iter):
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

data_train_ls, data_test_ls = [], []
train_text_ls, train_label_ls = list(data_train.Segmented_Text), list(data_train.Label)
test_text_ls, test_label_ls = list(data_test.Segmented_Text), list(data_test.Label)
for i in range(len(train_text_ls)):
      data_train_ls.append([train_text_ls[i],train_label_ls[i]])
for i in range(len(test_text_ls)):
      data_test_ls.append([test_text_ls[i],test_label_ls[i]])

batch_size = 64
vocabs = get_vocab(data_train_ls)
print('Words in Vocabulary: {}'.format(len(vocabs)))
train_set = Data.TensorDataset(*preprocess(data_train_ls, vocabs))
test_set = Data.TensorDataset(*preprocess(data_test_ls, vocabs))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)


class GlobalMaxPool1d(nn.Module): # 用一维池化层实现时序最大池化层
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, channel, seq_len)
        # return shape: (batch_size, channel, 1)
        return F.max_pool1d(x, kernel_size=x.shape[2])


class TextCNN(nn.Module):
    def __init__(self, vocab, embed_size, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # 不参与训练的嵌入层
        self.constant_embedding = nn.Embedding(len(vocab), embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 2)
        # 时序最大池化层没有权重，所以可以共用一个实例
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2 * embed_size,
                                        out_channels=c,
                                        kernel_size=k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)), dim=2)  # (batch, seq_len, 2*embed_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        encoding = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        # 应用丢弃法后使用全连接层得到输出
        outputs = self.decoder(self.dropout(encoding))
        return outputs


embed_size, kernel_sizes, nums_channels = 300, [3, 4, 5], [300, 300, 300]
net = TextCNN(vocabs, embed_size, kernel_sizes, nums_channels)

cache = '.vector_cache'
if not os.path.exists(cache):
    os.mkdir(cache)
glove_vocab = Vocab.Vectors(name='data/sgns.baidubaike.bigram-char', cache=cache)
net.embedding.weight.data.copy_(
    load_pretrained_embedding(vocabs.itos, glove_vocab))
net.constant_embedding.weight.data.copy_(
    load_pretrained_embedding(vocabs.itos, glove_vocab))
net.constant_embedding.weight.requires_grad = False

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

