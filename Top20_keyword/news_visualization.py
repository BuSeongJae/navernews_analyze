import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from matplotlib import font_manager, rc

font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)


train_data =pd.read_csv(r'C:\Users\user\Desktop\db_data\news(1028)\news_traindata.csv',encoding='utf-8')

train_data.columns = ['no', 'news_no','title','company', 'section', 'date', 'rank', 'contents']


train_data.columns = ['no', 'news_no','title','company', 'section', 'date', 'rank', 'contents']


train_data.drop_duplicates(subset=['title'], inplace=True)

train_data = train_data.dropna(how = 'any') # Null 값이 존재하는 행 제거

train_data.drop_duplicates(subset=['title'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거

nouns = []
okt = Okt()
for sentence in tqdm(train_data['title']):
    tokenized_sentence = okt.nouns(sentence)  # 토큰화
    nouns += tokenized_sentence
    for i, v in enumerate(nouns):
        if len(v) < 2:
            nouns.pop(i)

from collections import Counter
count = Counter(nouns)


nouns_list=[]
nouns_list = count.most_common(100)
for v in nouns_list:
    print(v)

words = []
count = []

for i in range(20):
    words += nouns_list[i][0]
    count += str(nouns_list[i][1])

words = [nouns_list[i][0] for i in range(20)]
count = [str(nouns_list[i][1]) for i in range(20)]
count = list(map(int, count))
words.reverse()
count.reverse()


plt.barh(words, count)
plt.title('뉴스 키워드 상위 Top 20')
plt.show()




