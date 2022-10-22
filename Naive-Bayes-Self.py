from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random
import jieba

total_positive_line = 500
total_negative_line = 500
dic_pos_text,dic_neg_text={},{}
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
            data.append([word_ls, 1])

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
            data.append([word_ls, 0])

    print('Positive Line: {} | Negative Line: {}\n'.format(positive_line,negative_line))
    random.shuffle(data)
    return data

def data_info(data_train,data_test):
      all_training_words = [word for tokens,_ in data_train for word in tokens]
      training_sentence_lengths = [len(tokens) for tokens,_ in data_train]
      TRAINING_VOCAB = sorted(list(set(all_training_words)))
      print("TRAIN DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}".format(len(all_training_words),len(TRAINING_VOCAB),max(training_sentence_lengths)))

      all_test_words = [word for tokens,_ in data_test for word in tokens]
      test_sentence_lengths = [len(tokens) for tokens,_ in data_test]
      TEST_VOCAB = sorted(list(set(all_test_words)))
      print("TEST  DATASET | Total Words:{} | Total Vocabulary:{}  | Max Sentence Length:{}\n" .format(len(all_test_words),len(TEST_VOCAB),max(test_sentence_lengths)))

def vocab_dict(data_train):   #Setting up Vocabulary Dict
      print('Setting up Vocabulary Dict...')
      vocab = collections.Counter()
      for twit,label in data_train:
            for word in twit:
                  vocab[word] += 1  
      print(len(vocab),'Vocabularies\n')
      vocab_ls = list(vocab)
      return vocab, vocab_ls

def word2vec(data_train=None,data_test=None,vocab_ls=None,test_ls=None):
      vocab_len = len(vocab_ls)

      #Test individual sentence Word2vec
      if not test_ls == None:
            word2vec_st = np.zeros([1,vocab_len])
            for i in range(len(test_ls)):
                  for j in range(vocab_len):
                        if test_ls[i] == vocab_ls[j]:
                              word2vec_st[0][j] = 1
            return word2vec_st

      #Train Dataset Word2vec
      length = len(data_train)
      word2vec_train = np.zeros([length,vocab_len])
      k = 0
      for twit,label in data_train:
            k += 1
            if k == length:
                  end_val = ' | '
            else:
                  end_val = '\r'
            print('Getting Train Dataset Word2vec: {}/{}'.format(k,length),end=end_val)
            for i in range(len(twit)):
                  for j in range(vocab_len):
                        if twit[i] == vocab_ls[j]:
                              #word2vec_train[k-1][j] = 1*vocab[vocab_ls[j]]/vocab_len
                              word2vec_train[k-1][j] = 1
      print('Vector Shape:',word2vec_train.shape)

      #Test Dataset Word2vec
      length = len(data_test)
      word2vec_test = np.zeros([length,vocab_len])
      k = 0
      for twit,label in data_test:
            k += 1
            if k == length:
                  end_val = ' | '
            else:
                  end_val = '\r'
            print('Getting Test Dataset Word2vec: {}/{}'.format(k,length),end=end_val)
            for i in range(len(twit)):
                  for j in range(vocab_len):
                        if twit[i] == vocab_ls[j]:
                              #word2vec_test[k-1][j] = 1*vocab[vocab_ls[j]]/vocab_len
                              word2vec_test[k-1][j] = 1
      print('Vector Shape:',word2vec_test.shape,'\n')

      label_train = [label for tokens,label in data_train]
      label_test = [label for tokens,label in data_test]
      X_train, X_test = word2vec_train, word2vec_test
      y_train, y_test = np.array(label_train), np.array(label_test)
      return X_train, X_test, y_train, y_test


def get_pos_neg_dic(data_train):
      for i in range(len(data_train)):
            if data_train[i][1] == 1:
                  for j in range(len(data_train[i][0])):      
                        if data_train[i][0][j] in dic_pos_text:
                              dic_pos_text[data_train[i][0][j]] += 1
                        else:
                              dic_pos_text[data_train[i][0][j]] = 1
            else:
                  for j in range(len(data_train[i][0])):  
                        if data_train[i][0][j] in dic_neg_text:
                              dic_neg_text[data_train[i][0][j]] += 1
                        else:
                              dic_neg_text[data_train[i][0][j]]= 1

def jieba_segment(text):
    word_ls = []
    tmp_ls = list(jieba.cut(text, cut_all=False))   #segmentation
    for i in range(len(tmp_ls)):
        if not isStopWord(tmp_ls[i]):
            word_ls.append(tmp_ls[i]) 
    return word_ls

def Bayes(X_data=None,y_data=None,test_word2vec=None,output=True):
      if not test_word2vec == None:
            train_positive_line = list(test_word2vec).count(1)
            train_negative_line = list(test_word2vec).count(0)
            prob_c_pos = train_positive_line/(train_negative_line+train_positive_line)
            prob_c_neg = train_negative_line/(train_negative_line+train_positive_line)
            prob_xi_c_pos,prob_xi_c_neg = 1,1
            for j in range(len(test_word2vec)):
                  if(test_word2vec[j] == 1):
                        real_word = vocab_ls[j]
                        if real_word in dic_pos_text:
                              if not train_positive_line == 0:
                                    prob_xi_c_pos *= (dic_pos_text[real_word] / train_positive_line * prob_c_pos)
                        if real_word in dic_neg_text:
                              if not train_negative_line == 0:
                                    prob_xi_c_neg *= (dic_neg_text[real_word] / train_negative_line *prob_c_neg)
            prob_c_x_pos = prob_xi_c_pos * prob_c_pos 
            prob_c_x_neg = prob_xi_c_neg * prob_c_neg 
            if not prob_c_x_pos+prob_c_x_neg == 0:
                  prob_pos = (1 - prob_c_x_pos/(prob_c_x_pos+prob_c_x_neg)) * 100
                  prob_neg = (1 - prob_c_x_neg/(prob_c_x_pos+prob_c_x_neg)) * 100
            if prob_c_x_pos <= prob_c_x_neg: 
                  guess_label = 1
            else:
                  guess_label = 0
            if output:
                  print('Positive Probability = {:.4f}%'.format(prob_pos))
                  print('Negative Probability = {:.4f}%'.format(prob_neg))
            return guess_label

      guess_y = []
      train_positive_line = list(y_data).count(1)
      train_negative_line = list(y_data).count(0)
      prob_c_pos = train_positive_line/(train_negative_line+train_positive_line)
      prob_c_neg = train_negative_line/(train_negative_line+train_positive_line)
      for vector in X_data:
            prob_xi_c_pos,prob_xi_c_neg = 1,1
            for j in range(len(vector)):
                  if(vector[j] == 1):
                        real_word = vocab_ls[j]
                        if real_word in dic_pos_text:
                              if not train_positive_line == 0:
                                    prob_xi_c_pos *= (dic_pos_text[real_word] / train_positive_line * prob_c_pos)
                        if real_word in dic_neg_text:
                              if not train_negative_line == 0:
                                    prob_xi_c_neg *= (dic_neg_text[real_word] / train_negative_line *prob_c_neg)
            prob_c_x_pos = prob_xi_c_pos * prob_c_pos
            prob_c_x_neg = prob_xi_c_neg * prob_c_neg 

            if prob_c_x_pos <= prob_c_x_neg: 
                  guess_label = 1
            else:
                  guess_label = 0
            guess_y.append(guess_label)

      correct_count = 0
      for i in range(len(y_data)):
            if y_data[i] == guess_y[i]:
                  correct_count += 1
      print('Accuracy: {}'.format(correct_count/(train_negative_line+train_positive_line)),end='')
      return 


def test_sample_data(vocab_ls):
      print('\n\n')
      text = ["房间设施难以够得上五星级，服务还不错，有送水果。",
            "前台服务较差，不为客户着想。房间有朋友来需要打扫，呼叫了两个小时也未打扫。房间下水道臭气熏天，卫生间漏水堵水。"]
      for i in range(len(text)):
            print('Test Sample '+str(i)+'\n'+text[i])   
            tokens = jieba_segment(text[i])
            word2vec_st = word2vec(None,None,vocab_ls=vocab_ls,test_ls=tokens)
            sentiment = Bayes(None,None,list(word2vec_st[0]))
            if sentiment == 1:
                  print("Positive Sentiment 正面情绪\n")
            else:
                  print("Negative Sentiment 负面情绪\n")

if __name__ == '__main__':   
      data_train, data_test = train_test_split(load_data(), test_size=0.2)
      data_info(data_train,data_test)
      vocab, vocab_ls = vocab_dict(data_train)
      X_train, X_test, y_train, y_test = word2vec(data_train=data_train, data_test=data_test, vocab_ls=vocab_ls)
      get_pos_neg_dic(data_train)
      print('\nTRAINING MODEL')
      print('Train',end=' ')
      Bayes(X_train,y_train)  #Train data
      print(' | Test',end=' ')
      Bayes(X_test,y_test)    #Test data
      test_sample_data(vocab_ls)
