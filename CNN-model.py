import collections
from ctypes import sizeof
import os
import random
import time
import jieba
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torchtext.legacy import vocab
import torchtext.vocab as Vocab
import torch.utils.data as Data
from tqdm import tqdm
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_size =500
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

def load_data():   #获取数据集
    data = []
    getStopWord()
    positive_line = 0
    negative_line = 0
    total_positive_line = data_size
    total_negative_line = data_size

    with open('data/total/pos.txt', 'r', encoding='utf-8') as f:
        print('File Directory: data/total/pos.txt')
        sentences = f.readlines()
        for sentence in sentences[:total_positive_line]:
            positive_line += 1
            if positive_line == total_positive_line:
                end_val = '\n'
            else:
                end_val = '\r'
            print('Getting positive sentence {}/{}'.format(positive_line,total_positive_line),end=end_val)
            word_ls = []
            words = sentence.replace('\n','').split('    ')   #get chinese sentence
            tmp_ls = list(jieba.cut(words[1], cut_all=True))   #segmentation
            for i in range(len(tmp_ls)):
                if not isStopWord(tmp_ls[i]):
                       word_ls.append(tmp_ls[i]) 
            data.append([word_ls, 0])

    with open('data/total/neg.txt', 'r', encoding='utf-8') as f:
        print('File Directory: data/total/neg.txt')
        sentences = f.readlines()
        for sentence in sentences[:total_negative_line]:
            negative_line += 1
            if negative_line == total_negative_line:
                end_val = '\n'
            else:
                end_val = '\r'
            print('Getting negative sentence {}/{}'.format(negative_line,total_negative_line),end=end_val)
            word_ls = []
            words = sentence.replace('\n','').split('    ')   #get chinese sentence
            tmp_ls = list(jieba.cut(words[1], cut_all=True))   #segmentation
            for i in range(len(tmp_ls)):
                if not isStopWord(tmp_ls[i]):
                       word_ls.append(tmp_ls[i]) 
            data.append([word_ls, 1])

    print('Positive Line: {} | Negative Line: {}'.format(positive_line,negative_line))
    random.shuffle(data)
    return data


def get_vocab(data):
    tokenized_data = [words for words, _ in data]
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return vocab.Vocab(counter, min_freq=5)

def data_info(train_data,test_data):
    all_training_words = [word for lines in train_data for word in lines[0]]
    training_sentence_lengths = [len(lines) for lines in train_data]
    TRAINING_VOCAB = sorted(list(set(all_training_words)))
    print("TRAIN DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}".format(len(all_training_words),len(TRAINING_VOCAB),max(training_sentence_lengths)))

    all_test_words = [word for lines in test_data for word in lines[0]]
    test_sentence_lengths = [len(tokens) for tokens in train_data]
    TEST_VOCAB = sorted(list(set(all_test_words)))
    print("TEST  DATASET | Total Words:{}  | Total Vocabulary:{} | Max Sentence Length:{}" .format(len(all_test_words),len(TEST_VOCAB),max(test_sentence_lengths)))

def preprocess(data, vocab):  #data:list | vocab:torchtext.legacy.vocab.Vocab
    max_l = 500  
    def set_same_length(x):   #将每条评论通过截断或者补0，使得长度变成500
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = [words for words, _ in data]     
    features = torch.tensor([set_same_length([vocab.stoi[word] for word in words]) for words in tokenized_data])
    #stoi: A collections.defaultdict instance mapping token strings to numerical identifiers.
    labels = torch.tensor([score for _, score in data])
    #features.shape 为 ([len(tokenized_data),500])
    #features.shape 为 ([len(tokenized_data)])
    return features, labels


def pretrain_embedding(words, glove_vocab):  #vocab.itos, glove_vocab; glove_vocab: vector representations for words.
    """从预训练好的vocab中提取出words对应的词向量"""
    #len(words) = len(vocab.itos) = len(vocab)
    #glove_vocab.vectors[0].shape 为 ([300])
    #embed.shape 为 ([len(vocab),300])
    embedding = torch.zeros(len(words), glove_vocab.vectors[0].shape[0])  # 初始化为0  
    
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = glove_vocab.stoi[word]
            embedding[i, :] = glove_vocab.vectors[idx]   #大小为([300]) 的torch数组
        except KeyError:    #单词不在字典里
            oov_count += 1
    if oov_count > 0:
        print("Words Out of Vocabulary: {}".format(oov_count))
    return embedding  #大小 为 ([len(vocab),300])，作为模型的参数矩阵

class GlobalMaxPool1d(nn.Module): # 用一维池化层实现时序最大池化层
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return nn.functional.max_pool1d(x, kernel_size=x.shape[2])

class CNN(nn.Module):
    def __init__(self, vocab, embedding_size, kernel_sizes, num_channels):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_size)
        self.constant_embedding = nn.Embedding(len(vocab), embedding_size) # 不参与训练的嵌入层
        #print(len(vocab),embedding_size)  | 1392,300
        self.dropout = nn.Dropout(0.5)
        self.feed_forward = nn.Linear(sum(num_channels), 2) #创建具有 900 个输入和 2 个输出的单层前馈网络; sum(num_channels)=900
        self.pool = GlobalMaxPool1d() # 时序最大池化层没有权重，所以可以共用一个实例
        self.convs = nn.ModuleList()  # 创建多个一维卷积层
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels=2*embedding_size, out_channels=c, kernel_size=k))

    def forward(self, inputs):
        # 将两个形状是(批量大小, 词数, 词向量维度)的嵌入层的输出按词向量连结
        embeddings = torch.cat((
            self.embedding(inputs),
            self.constant_embedding(inputs)), dim=2)  # (batch, seq_len, 2*embedding_size)
        # 根据Conv1D要求的输入格式，将词向量维，即一维卷积层的通道维(即词向量那一维)，变换到前一维
        embeddings = embeddings.permute(0, 2, 1)
        
        # 对于每个一维卷积层，在时序最大池化后会得到一个形状为(批量大小, 通道大小, 1)的
        # Tensor。使用flatten函数去掉最后一维，然后在通道维上连结
        conv_ls = []
        for conv in self.convs:
            c = conv(embeddings)   
            r = nn.functional.relu(c)
            p = self.pool(r)
            s = p.squeeze(-1)
            conv_ls.append(s)
        cnn_layers = torch.cat(conv_ls,dim=1)
        # 应用丢弃法后使用全连接层得到输出
        d = self.dropout(cnn_layers)
        outputs = self.feed_forward(d)
        return outputs

def train(train_iteration, test_iteration, model, loss, optimizer, device, num_epochs):
    model = model.to(device)
    
    print("training on ", device)
    batch_count = 0
    train_acc_ls,test_acc_ls,loss_ls = [],[],[]

    for epoch in range(num_epochs):  #迭代次数
        train_loss_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in tqdm(train_iteration):    #creating Progress Meters or Progress Bars，训练在cpu执行
            '''迭代训练集中的所有样本'''
            X = X.to(device)    #DataLoader works on CPU and only after the batch is retrieved data is moved to GPU.
            y = y.to(device)
            y_hat = model(X)      #估计标签
            loss_func = loss(y_hat, y)   #cross entropy loss
            optimizer.zero_grad()
            loss_func.backward()   #反向计算误差
            optimizer.step()  #根据梯度更新参数，梯度是optimizer中的parameter
            train_loss_sum += loss_func.cpu().item()  
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()   # 估计标签=真正标签 的个数
            n += y.shape[0]
            batch_count += 1
        test_acc = test_accuracy(test_iteration, model)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'%(epoch+1, train_loss_sum/batch_count, train_acc_sum/n, test_acc, time.time()-start))

        train_acc_ls.append(train_acc_sum / n)
        test_acc_ls.append(test_acc)
        loss_ls.append(train_loss_sum / batch_count)
    return train_acc_ls,test_acc_ls,loss_ls

def test_accuracy(data_iter, model, device=None):
    if device is None and isinstance(model, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(model.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            model.eval()  # 评估模式, 这会关闭dropout (on CPU)
            acc_sum += (model(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()  # 估计标签=真正标签 的个数
            model.train()  # 改回训练模式 
            n += y.shape[0]
    return acc_sum / n

def get_figure(num_epochs,loss_ls,train_acc_ls,test_acc_ls):
    x = [*range(1, num_epochs+1, 1)]
    plt.plot(x, loss_ls, label='loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

    plt.plot(x, train_acc_ls, label='train accuracy')
    plt.plot(x, test_acc_ls, label='test accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()



if __name__ == '__main__':

    #get data and vocab
    batch_size = 64
    train_data, test_data = train_test_split(load_data(), test_size=0.2)
    data_info(train_data,test_data)
    vocab = get_vocab(train_data)
    print('Words in Vocabulary: {}'.format(len(vocab)))

    #setup data, get features and labels
    train_set = Data.TensorDataset(*preprocess(train_data, vocab))
    test_set = Data.TensorDataset(*preprocess(test_data, vocab))
    train_iteration = Data.DataLoader(train_set, batch_size, shuffle=True)  #data for Progress Meters or Progress Bars
    test_iteration = Data.DataLoader(test_set, batch_size)

    #setup CNN model
    embedding_size, kernel_sizes, num_channel = 300, [2, 3, 4], [300, 300, 300]
    model = CNN(vocab, embedding_size, kernel_sizes, num_channel)

    #Word2vec
    cache = '.vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    glove_vocab = Vocab.Vectors(name='data/sgns.baidubaike.bigram-char', cache=cache)  
    #GloVe: an unsupervised learning algorithm for obtaining vector representations for words.

    #setup weight of CNN model
    model.embedding.weight.data.copy_(pretrain_embedding(vocab.itos, glove_vocab))
    model.constant_embedding.weight.data.copy_(pretrain_embedding(vocab.itos, glove_vocab))
    model.constant_embedding.weight.requires_grad = False

    #train model
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()
    train_acc_ls,test_acc_ls,loss_ls = train(train_iteration, test_iteration, model, loss, optimizer, device, num_epochs)

    #get loss and accuracy figure
    get_figure(num_epochs,loss_ls,train_acc_ls,test_acc_ls)

    print(model)



