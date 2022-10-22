#朴素贝叶斯分词(含义是分词后，得分的假设是基于两词之间是独立的，后词的出现与前词无关)
# p[i][n]表示从i到n的句子的最佳划分的得分,我们用dp表达式p[i][n]=max(freq(s[i:k])+p[k][n])
# 依次求出长度为1,2,3,n的句子划分，那么p[0][n]便是最佳划分结果,用t[i]表示产生的最佳划分每次向前走几个字符

import math
from math import log, exp
import xlwt
import xlrd
from xlutils.copy import copy
d = {}
log = lambda x: float('-inf') if not x else math.log(x)    
prob = lambda x: d[x][0] if x in d else 0 if len(x)>1 else 1
getpos = lambda x: d[x][1] if x in d else 0 if len(x)>1 else 1

stopword_ls = []
def getStopWord():
      with open('lib/stopwords_utf8.txt', 'r',encoding='UTF-8') as file:
            for line in file:
                  stopword_ls.append(line.split('\n')[0])

class Bayes:
      def __init__(self,file_dir=None):

            self.ls = []
            self.ls.append(file_dir)
            d['_N_'] = 0.0
            with open('lib/SogouLabDic.dic','r',encoding='UTF-8') as file:
                  for line in file:
                        #print(line)
                        word, freq ,pos= line.split('\t')[0:3]  #取list的前3个元素,词和相应的词数
                        d['_N_'] += int(freq)+1                  # 此表的词频总和,每个词数都加1    
                        try:
                              d[word.decode('utf-8')] = [int(freq)+1,pos] #词数加1
                        except:
                              try:
                                    d[word] = [int(freq)+1,pos]            #词数加1
                              except:
                                    print(word)
                                    break

      def segment(self,s='',output=False):
            self.ls.append(s)
            if output:
                  print('Before word segmentation:',s)     
            l = len(s)
            p = [0 for i in range(l+1)] #1,2,...,l位置为0
            t = [0 for i in range(l)]
            word = [0 for i in range(l+1)]
            pos = [0 for i in range(l+1)]
            raw_word = []
            final_word = []
            
            #t[i]保留从当前位置向前划分的最佳长度，，取决词库
            for i in range(l-1, -1, -1): #start,stop,step
                  # prob(s[i:i+k])/d['_t_'] 为词表词频度
                  p[i], t[i],word[i],pos[i] = max((prob(s[i:i+k])/d['_N_']+p[i+k], k, s[i:i+k], getpos(s[i:i+k]))#在一个二元组列表里返回第一个元素最大的二元组,
                         for k in range(1,l-i+1))              
                  #print('[{}, {:.8f}, {}]'.format(word[i],p[i],t[i]))

            dis = 0
            while dis<l:  #dis=0,不断向前遍历分割词汇
                  #yield s[dis:dis+t[dis]]
                  #print (s[dis:dis+t[dis]])
                  final_word.append(s[dis:dis+t[dis]])
                  dis += t[dis]
            
            self.ls.append(" | ".join(final_word))
            if output:
                  print('After word segmentation:',final_word)
            self.final_word = final_word
            self.raw_word = raw_word

      def sentimental(self,output=False):
            final_word = self.final_word
            raw_word = self.raw_word
            positive_count = 0
            negative_count = 0
            sentimental_score = 0.0
            is_stopword = False
            positive_ls = []
            negative_ls = []

            for i in range(len(final_word)):
                  is_stopword =  False
                  for j in range(len(stopword_ls)):
                        if final_word[i] == stopword_ls[j]: 
                              is_stopword = True
                  if not is_stopword:
                        raw_word.append(final_word[i])
            #print('Raw word:',raw_word)   
            self.total_word_count = len(final_word)

            #POSITIVE WORD
            with open('lib/full_pos_dict_sougou.txt', 'r',encoding='UTF-8') as file:
                  for line in file:
                        line_adj = line.split('\n')[0]
                        for j in range(len(raw_word)):
                              if raw_word[j] == line_adj:
                                    #print('positive:',raw_word[j])
                                    positive_count += 1
                                    positive_ls.append(raw_word[j])

            #NEGATIVE WORD
            with open('lib/full_neg_dict_sougou.txt', 'r',encoding='UTF-8') as file:
                  for line in file:
                        line_adj = line.split('\n')[0]
                        for j in range(len(raw_word)):
                              if raw_word[j] == line_adj:
                                    #print('negative:',raw_word[j])
                                    negative_count += 1
                                    negative_ls.append(raw_word[j])

            #Caluculating sentimental score using Positive and Negative Word Counts          
            #sentimental_score = round(positive_count/(negative_count+1), 2)        #With Semi Normalization
            if not len(raw_word)==0 :
                  sentimental_score = round((positive_count-negative_count)/len(raw_word), 2)  #With Normalization                 
            if output:
                  print('Positive Count: {} | Negative Count: {}'.format(positive_count,negative_count))
                  print('Sentimental Score: {}'.format(sentimental_score))    
                  print('-'*84)   
            self.positive_count = positive_count
            self.negative_count = negative_count
            self.sentimental_score = sentimental_score
            self.ls.append(positive_count)
            self.ls.append(", ".join(positive_ls))
            self.ls.append(negative_count)
            self.ls.append(", ".join(negative_ls))
            self.ls.append(sentimental_score)               
        
class getFile:
      def __init__(self,sheetname):
            self.sheetname = sheetname

      def set_excel(self):
            wb = xlwt.Workbook()              # 创建 worksheet
            ws = wb.add_sheet(self.sheetname)  # 创建工作表
            for i in range (8):
                  ws.col(i).width = 3500        # 设置每列的宽度，方便用户浏览
            style = xlwt.easyxf('pattern: pattern solid;''font: colour orange, bold True, height 280;')  #设置字样和格式
            default_style = xlwt.easyxf('font: colour black, bold True, height 225;')
            ws.write(0, 0, '朴素贝叶斯中文分词（'+self.sheetname+'）:', style=style)   # 写入第一行标题  ws.write(a, b, c)  a：行，b：列，c：内容
            ws.write(1, 0, '文档名', style=default_style)
            ws.write(1, 1, '分词前', style=default_style)
            ws.write(1, 2, '分词后', style=default_style)
            ws.write(1, 3, '正面词语个数', style=default_style)
            ws.write(1, 4, '正面词语', style=default_style)
            ws.write(1, 5, '负面词语个数', style=default_style)   
            ws.write(1, 6, '负面词语', style=default_style)
            ws.write(1, 7, '情绪评分', style=default_style)
            wb.save('result/'+self.sheetname+'-分词和情绪分析.xls')

      def save_to_excel(self,ls):
            workbook = xlrd.open_workbook('result/'+self.sheetname+'-分词和情绪分析.xls', formatting_info=True)
            sheet = workbook.sheet_by_index(0)
            rowNum = sheet.nrows
            newbook = copy(workbook)
            newsheet = newbook.get_sheet(0)	
            for i in range(len(ls)):
                  newsheet.write(rowNum, i, ls[i])
            newbook.save('result/'+self.sheetname+'-分词和情绪分析.xls')     
      


if __name__ == '__main__':
      getOutput = False
      getStopWord()
      data_set = 0 #sentence
      #data_set = 1 #paragraph
      dataset_size = 1000
      print('---------------Using Naive Bayes Realizing Chinese Word Segmentation---------------')
      if data_set == 0:   
            f1 = getFile('Positive1')
            f1.set_excel()
            accurate_count = 0
            for i in range(dataset_size):
                  with open('data/positive/pos.'+str(i)+'.txt','r',encoding='UTF-8') as file:
                        text = file.read().replace('\n', '')
                  print('File directory: pos.'+str(i)+'.txt')
                  b = Bayes(file_dir='pos.'+str(i)+'.txt')
                  b.segment(text,output=getOutput)
                  b.sentimental(output=getOutput)
                  f1.save_to_excel(b.ls)
                  if b.sentimental_score >= 0:
                        accurate_count += 1
            print('Accuracy for Positive Dataset: {}'.format(accurate_count/dataset_size))

            f2 = getFile('Negative1')
            f2.set_excel()
            for i in range(dataset_size):
                  with open('data/negative/neg.'+str(i)+'.txt','r',encoding='UTF-8') as file:
                        text = file.read().replace('\n', '')
                  print('File directory: neg.'+str(i)+'.txt')
                  b = Bayes(file_dir='neg.'+str(i)+'.txt')
                  b.segment(text,output=getOutput)
                  b.sentimental(output=getOutput)
                  f2.save_to_excel(b.ls)
                  if b.sentimental_score >= 0:
                        accurate_count += 1
            print('Accuracy for Negative Dataset: {}'.format(accurate_count/dataset_size))

      elif data_set == 1:   #Whole Paragraph
            #POSITIVE PARAGRAPH
            line = 0
            total_line = 7000
            positive_count = 0
            negative_count = 0
            total_word_count = 0
            print('File directory: data/total/pos.txt')
            with open('data/total/pos.txt', 'r', encoding='utf-8') as f:
                  sentences = f.readlines()
                  for sentence in sentences[:total_line]:
                        line += 1
                        if line == total_line:
                              end_val = '\n'
                        else:
                              end_val = '\r'
                        print('Getting sentence {}/{}'.format(line,total_line),end=end_val)
                        words = sentence.strip().replace(' ','').replace('1','').split('\t')
                        b = Bayes()
                        b.segment(s=words[0],output=False)
                        b.sentimental(output=False)
                        positive_count += b.positive_count
                        negative_count += b.negative_count
                        total_word_count += b.total_word_count

            sentimental_score = round((positive_count-negative_count)/total_word_count, 2)  #With Normalization
            print('Total Word Count: {}'.format(total_word_count),end='\n')
            print('Positive Count: {} | Negative Count: {}'.format(positive_count,negative_count))
            print('Sentimental Score: {}'.format(sentimental_score))    
            print('-'*84)

            #NEGATIVE PARAGRAPH
            line = 0
            total_line = 3000
            positive_count = 0
            negative_count = 0
            total_word_count = 0
            print('File directory: data/total/neg.txt')
            with open('data/total/neg.txt', 'r', encoding='utf-8') as f:
                  sentences = f.readlines()
                  for sentence in sentences[:total_line]:
                        line += 1
                        if line == total_line:
                              end_val = '\n'
                        else:
                              end_val = '\r'
                        print('Getting sentence {}/{}'.format(line,total_line),end=end_val)
                        words = sentence.strip().replace(' ','').replace('-1','').split('\t')
                        b = Bayes()
                        b.segment(s=words[0],output=False)
                        b.sentimental(output=False)
                        positive_count += b.positive_count
                        negative_count += b.negative_count
                        total_word_count += b.total_word_count

            sentimental_score = round((positive_count-negative_count)/total_word_count, 2)  #With Normalization
            print('Total Word Count: {}'.format(total_word_count),end='\n')
            print('Positive Count: {} | Negative Count: {}'.format(positive_count,negative_count))
            print('Sentimental Score: {}'.format(sentimental_score))    
            print('-'*84)
            
            
            

