from snownlp import SnowNLP
import random

def get_sentiment(sentiment,file_dir,range_int):
      if sentiment == 'both':
            print('-'*40,'COMBINE','-'*40)
            num1 = random.randint(0,range_int) 
            num2 = random.randint(0,range_int) 
            print('Positive Dataset:{} | Negative Dataset:{}'.format(num1,num2))
            st = ''

            #Positive
            line = 0
            print("File Directory: data/total/pos.txt")
            with open('data/total/pos.txt', 'r', encoding='utf-8') as f:
                  sentences = f.readlines()
                  for sentence in sentences[:num1]:
                        line+=1
                        if line == num1:
                              end_val = '\n'
                        else:
                              end_val = '\r'
                        print("Getting Positive sentence {}/{}".format(line,num1),end=end_val)
                        st += sentence
            #Negative
            line = 0
            print("File Directory: data/total/neg.txt")
            with open('data/total/neg.txt', 'r', encoding='utf-8') as f:
                  sentences = f.readlines()
                  for sentence in sentences[:num2]:
                        line+=1
                        if line == num2:
                              end_val = '\n'
                        else:
                              end_val = '\r'
                        print("Getting Negative sentence {}/{}".format(line,num2),end=end_val)
                        st += sentence
            s = SnowNLP(st)
            print("Getting Sentiment Analysis ...")
            print("Positive Sentiment Probability:",str(s.sentiments))
            return

      if sentiment == 'positive':
            print('-'*40,'POSITIVE','-'*40)
      elif sentiment == 'negative':
            print('-'*40,'NEGATIVE','-'*40)

      st = ''
      line = 0
      total_line = random.randint(0,range_int)
      print("File Directory:",file_dir)
      with open(file_dir, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
            for sentence in sentences[:total_line]:
                  line+=1
                  if line == total_line:
                        end_val = '\n'
                  else:
                        end_val = '\r'
                  print("Getting sentence {}/{}".format(line,total_line),end=end_val)
                  st += sentence
      s = SnowNLP(st)
      print("Getting Sentiment Analysis ...")
      print("Positive Sentiment Probability: {}".format(s.sentiments))

if __name__ == '__main__':
      range_int = 100
      get_sentiment('positive','data/total/pos.txt', range_int)
      get_sentiment('negative','data/total/neg.txt', range_int)
      get_sentiment('both',None, range_int)

