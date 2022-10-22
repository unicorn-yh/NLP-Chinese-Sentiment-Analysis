from sklearn.model_selection import train_test_split
import random
import jieba
from time import time
from nltk import classify
from nltk import NaiveBayesClassifier

stopword_ls = []
def getStopWord():
    with open('lib/stopwords_utf8.txt', 'r',encoding='UTF-8') as file:
        for line in file:
            stopword_ls.append(line.split('\n')[0])
getStopWord()

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
    total_positive_line = 7000
    total_negative_line = 3000

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

def list_to_dict(cleaned_tokens):
    return dict([token, True] for token in cleaned_tokens)

def data_info(data_train,data_test):
      all_training_words = [word for tokens,_ in data_train for word in tokens]
      training_sentence_lengths = [len(tokens) for tokens,_ in data_train]
      TRAINING_VOCAB = sorted(list(set(all_training_words)))
      print("TRAIN DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}".format(len(all_training_words),len(TRAINING_VOCAB),max(training_sentence_lengths)))

      all_test_words = [word for tokens,_ in data_test for word in tokens]
      test_sentence_lengths = [len(tokens) for tokens,_ in data_test]
      TEST_VOCAB = sorted(list(set(all_test_words)))
      print("TEST  DATASET | Total Words:{}  | Total Vocabulary:{} | Max Sentence Length:{}" .format(len(all_test_words),len(TEST_VOCAB),max(test_sentence_lengths)))

      final_train = [(list_to_dict(tokens),label) for tokens,label in data_train ]
      final_test = [(list_to_dict(tokens),label) for tokens,label in data_test ]
      return final_train,final_test

def Bayes(final_train,final_test):
      start_time = time()
      classifier = NaiveBayesClassifier.train(final_train)
      # Output the model accuracy on the train and test data
      print('Accuracy on train data: {:.4f}'.format(classify.accuracy(classifier, final_train)))
      print('Accuracy on test  data: {:.4f}'.format(classify.accuracy(classifier, final_test)))

      # Output Top 20 Sentiment Word
      print(classifier.show_most_informative_features(20))
      print('CPU Time:', time() - start_time)
      return classifier

def jieba_segment(text):
    word_ls = []
    tmp_ls = list(jieba.cut(text, cut_all=False))   #segmentation
    for i in range(len(tmp_ls)):
        if not isStopWord(tmp_ls[i]):
            word_ls.append(tmp_ls[i]) 
    return word_ls

def test_model(classifier):  #TEST DATASET
      print('---------Naive Bayes Test Data----------')
      dataset_size = 1000
      correct_count = 0
      type = ['pos','neg']
      for j in range(len(type)):
            positive_count,negative_count = 0,0
            if type[j] == 'pos':
                  st = 'positive'
            else:
                  st = 'negative'
            for i in range(dataset_size):
                  with open('data/'+st+'/'+type[j]+'.'+str(i)+'.txt','r',encoding='UTF-8') as file:
                        text = file.read().replace('\n', '')
                        if i == dataset_size-1:
                              end_val = '\n'
                        else:
                              end_val = '\r'

                        print('Getting '+st+' data: {}/{}'.format(i+1,dataset_size),end=end_val)

                  custom_tokens = jieba_segment(text)
                  sentiment = classifier.classify(dict([token, True] for token in custom_tokens))
                  if sentiment == 0:
                        #print("Positive Sentiment")
                        positive_count += 1
                  else:
                        #print("Negative Sentiment")
                        negative_count += 1

            print('Positive Count:',positive_count,end = ' | ')
            print('Negative Count:',negative_count)
            if type[j] == 'pos':
                  correct_count += positive_count
            else:
                  correct_count += negative_count
      print('Accuracy: {}'.format(correct_count/(2*dataset_size)))

if __name__ == '__main__':
      data_train, data_test = train_test_split(load_data(), test_size=0.10)
      final_train,final_test = data_info(data_train,data_test)
      classifier = Bayes(final_train,final_test)
      test_model(classifier)

