#Logistic-Regression.py

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import collections
from snownlp import SnowNLP
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
stopword_ls = []

def concate_files(file1, file2, final_file, dataset_size):
      data1 = data2 = ""
      with open(file1, 'r', encoding='utf-8') as fp:   #Reading data from file1
            sentences = fp.readlines()
            for sentence in sentences[:int(dataset_size)]:
                  data1 += sentence
      with open(file2, 'r', encoding='utf-8') as fp:   #Reading data from file2
            sentences = fp.readlines()
            for sentence in sentences[:int(dataset_size)]:
                  data2 += sentence
      data1 += data2                                  #To add the data of file2
      with open (final_file, 'w', encoding='utf-8') as fp:
            fp.write(data1)

def getStopWord():
    with open('lib/stopwords_utf8.txt', 'r',encoding='UTF-8') as file:
        for line in file:
            stopword_ls.append(line.split('\n')[0])

def isStopWord(word):
    for i in range(len(stopword_ls)):
        if word == stopword_ls[i]:
            return True
    return False

def snow_segment(text):
    word_ls = []
    tmp_ls = SnowNLP(text).words       #word segmentation  
    for i in range(len(tmp_ls)):
        if not isStopWord(tmp_ls[i]):   #remove stopwords
            word_ls.append(tmp_ls[i]) 
    return word_ls

def preprocess(data,dataset_size):
      random.shuffle(data['Text'])
      data['Segmented_Text'] = data['Text'].apply(lambda x: snow_segment(x))  #segmentation
      
      #Assign Label to each sentence
      PosNegLabel = []
      for label in data.Label:
            if label == -1:        #negative
                  PosNegLabel.append(0)
            elif label == 1:      #positive
                  PosNegLabel.append(1)
      data['Label']= PosNegLabel
      data = data[['Segmented_Text','Label']]

      #Output train dataset info
      print('-'*40,dataset_size,'DATASET','-'*40)
      data_train, data_test = train_test_split(data, test_size=0.20)
      all_training_words = [word for tokens in data_train["Segmented_Text"] for word in tokens]
      training_sentence_lengths = [len(tokens) for tokens in data_train["Segmented_Text"]]
      TRAINING_VOCAB = sorted(list(set(all_training_words)))
      print("TRAIN DATASET | Total Words:{} | Total Vocabulary:{} | Max Sentence Length:{}".format(len(all_training_words),len(TRAINING_VOCAB),max(training_sentence_lengths)))

      #Output test dataset info
      all_test_words = [word for tokens in data_test["Segmented_Text"] for word in tokens]
      test_sentence_lengths = [len(tokens) for tokens in data_test["Segmented_Text"]]
      TEST_VOCAB = sorted(list(set(all_test_words)))
      print("TEST  DATASET | Total Words:{}  | Total Vocabulary:{} | Max Sentence Length:{}" .format(len(all_test_words),len(TEST_VOCAB),max(test_sentence_lengths)))
      return data_train,data_test

def vocab_dict(data_train):   #Setting up Vocabulary Dict
      print('Setting up Vocabulary Dict...')
      vocab = collections.Counter()
      for twit in data_train.Segmented_Text:
            for word in twit:
                  vocab[word] += 1  
      print(len(vocab),'Vocabularies')
      vocab_ls = list(vocab)
      return vocab, vocab_ls

def word2vec(data_train,data_test,vocab_ls):
      vocab_len = len(vocab_ls)

      #Train Dataset Word2vec
      length = len(data_train.Segmented_Text)
      word2vec_train = np.zeros([length,vocab_len])
      k = 0
      for twit in data_train.Segmented_Text:
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
      length = len(data_test.Segmented_Text)
      word2vec_test = np.zeros([length,vocab_len])
      k = 0
      for twit in data_test.Segmented_Text:
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
      print('Vector Shape:',word2vec_test.shape)

      X_train, X_test = word2vec_train, word2vec_test
      y_train, y_test = data_train['Label'], data_test['Label']
      return X_train, X_test, y_train, y_test

def getModel(X_train, X_test, y_train, y_test, dataset_size):
      print('Importing Logistic Regression Model ...')
      log_model = LogisticRegression().fit(X_train,y_train)
      print('Coefficients:', log_model.coef_)  # regression coefficients
      print('Dataset size:',dataset_size,end=' | ')
      print('Train Accuracy Score: {}'.format(log_model.score(X_train, y_train)),end=' | ')
      print('Test Accuracy Score: {}\n'.format(log_model.score(X_test, y_test)))
      return log_model
      
class LogisticRegression_:
    def __init__(self, learning_rate=0.01, num_iterations=50000, fit_intercept=True, verbose=True):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    # function to define the Incercept value.
    def b_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))  # initially we set it as all 1's
        # then we concatinate them to the value of X, we don't add we just append them at the end.
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid(self, z):  
        return 1 / (1 + np.exp(-z))

    def loss(self, yp, y):  #loss function
        return (-y * np.log(yp) - (1 - y) * np.log(1 - yp)).mean()

    def fit(self, X, y): #train model
        # as said if we want our intercept term to be added we use fit_intercept=True
        if self.fit_intercept:
            X = self.b_intercept(X)

        self.W = np.zeros(X.shape[1])  # weights initialization of our Normal Vector, initially we set it to 0, then we learn it eventually
        for i in range(self.num_iterations): 
            z = np.dot(X, self.W)  # W * Xi
            yp = self.sigmoid(z)  # predict the values of Y based on W and Xi
            gradient = np.dot(X.T, (yp - y)) / y.size    #gradient is calculated form the error generated by our model
            self.W -= self.learning_rate * gradient  #update weight, use the new values for the next iteration
            z = np.dot(X, self.W)  #new W * Xi
            yp = self.sigmoid(z)
            loss = self.loss(yp, y)  #loss
            
            # as mentioned above if we want to print somehting we use verbose, so if verbose=True then our loss get printed
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')

    # this is where we predict the actual values 0 or 1 using round. anything less than 0.5 = 0 or more than 0.5 is 1
    def predict(self, X):
      if self.fit_intercept:
            X = self.b_intercept(X) 
      prob = self.sigmoid(np.dot(X, self.W))  #predict the probability values based on out generated W values out of all those iterations.
      print(prob)
      return prob.round()

def plotConfusion(log_model):
      cm = confusion_matrix(y_test, log_model.predict(X_test))
      fig, ax = plt.subplots(figsize=(8, 8))
      ax.imshow(cm)
      ax.grid(False)
      ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 1', 'Predicted 0'))
      ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 1', 'Actual 0'))
      ax.set_ylim(1.5, -0.5)
      for i in range(2):
            for j in range(2):
                  ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
      plt.show()

      
if __name__ == '__main__':
      #dataset_num = [250,500,1000,1500,2000]
      dataset_num = [500]
      for dataset_size in dataset_num:
            concate_files('data/total/pos.txt','data/total/neg.txt','data/total/labeled_text.txt',dataset_size)
            data = pd.read_csv('data/total/labeled_text.txt', header = None, delimiter='    ',encoding='utf-8',names=['Label', 'Text'])
            getStopWord()
            data_train, data_test = preprocess(data,dataset_size)
            vocab, vocab_ls = vocab_dict(data_train)
            X_train, X_test, y_train, y_test = word2vec(data_train, data_test, vocab_ls)
            log_model = getModel(X_train, X_test, y_train, y_test, dataset_size)
            print(classification_report(y_test, log_model.predict(X_test)))
            plotConfusion(log_model)

            


            '''model = LogisticRegression_(learning_rate=0.1, num_iterations=30000)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            print((preds == y_test).mean())'''
