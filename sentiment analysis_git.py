#%%
1.   부도기업/우량기업 별 뉴스 데이터에 대한 이해와 전처리
2.   LSTM으로 네이버 뉴스데이터 감성 분류하기
3.   뉴스데이터 예측해보기
# %%
pip install konlpy
# %%
pip install --user tensorflow
# %%
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# %%
'''
[1] 부도기업/우량기업 별 뉴스 데이터에 대한 이해와 전처리
1. 데이터 로드
'''
# %%
targetDir_end = r"/Users/seonjin/ColabProjects/end_6m"
targetDir_end
targetDir_end2 = r"/Users/seonjin/ColabProjects/end_6m_2"
targetDir_end2
# %%
# targetDir_live = r"/Users/seonjin/ColabProjects/live"
targetDir_live = r"/Users/seonjin/ColabProjects/live_15e11"
targetDir_live
targetDir_live2 = r"/Users/seonjin/ColabProjects/live_15e11_2"
targetDir_live2
# %%
targetDir_mng = r"/Users/seonjin/ColabProjects/mng_6m"
targetDir_mng
#%%
##targetDir에서 .csv파일 이름들 리스트로 가져오기
file_list = os.listdir(targetDir_end)
csv_end_list = []
for file in file_list:
    if '.csv' in file:
        csv_end_list.append(file)
# %%
file_list = os.listdir(targetDir_end2)
csv_end_list2 = []
for file in file_list:
    if '.csv' in file:
        csv_end_list2.append(file)
# %%
file_list = os.listdir(targetDir_live)
csv_live_list = []
for file in file_list:
    if '.csv' in file:
        csv_live_list.append(file)
# %%
file_list = os.listdir(targetDir_live2)
csv_live_list2 = []
for file in file_list:
    if '.csv' in file:
        csv_live_list2.append(file)
# %%
file_list = os.listdir(targetDir_mng)
csv_mng_list = []
for file in file_list:
    if '.csv' in file:
        csv_mng_list.append(file)
# %%
end_list = pd.read_csv(targetDir_end + '/' + csv_end_list[0])
for csv_file in csv_end_list[1:]:
    target_path = targetDir_end + '/' + csv_file
    target_csv = pd.read_csv(target_path)
    end_list = end_list.append(target_csv) 
# %%
for csv_file in csv_end_list2[:]:
    target_path = targetDir_end2 + '/' + csv_file
    target_csv = pd.read_csv(target_path)
    end_list = end_list.append(target_csv) 
# %%
end_list['부도여부'] = 1
# %%
live_list = pd.read_csv(targetDir_live + '/' + csv_live_list[0])
for csv_file in csv_live_list[1:]:
    target_path = targetDir_live + '/' + csv_file
    target_csv = pd.read_csv(target_path)
    live_list = live_list.append(target_csv) 
# %%
for csv_file in csv_live_list2[:]:
    target_path = targetDir_live2 + '/' + csv_file
    target_csv = pd.read_csv(target_path)
    live_list = live_list.append(target_csv) 
# %%
live_list['부도여부'] = 0
# %%
mng_list = pd.read_csv(targetDir_mng + '/' + csv_mng_list[0])
for csv_file in csv_mng_list[1:]:
    target_path = targetDir_mng + '/' + csv_file
    target_csv = pd.read_csv(target_path)
    mng_list = mng_list.append(target_csv) 
# %%
mng_list['부도여부'] = 1
# %%
all_list = end_list[['뉴스','회사','부도여부']].append(live_list[['뉴스','회사','부도여부']])
all_list = pd.concat([live_list, end_list, mng_list], ignore_index=True)
all_list.shape
# (1288100, 4)
# %%
del end_list, live_list, mng_list
# %%
'''
2. 데이터 정제
'''
# %%
# 중복확인
all_list['뉴스'].nunique()
# %%
# 중복제거
all_list.drop_duplicates(subset=['뉴스'], inplace=True)
# 742350
# %%
all_list['부도여부'].value_counts().plot(kind = 'bar')
# %%
# 긍정 부정 개수
print(all_list.groupby('부도여부').size().reset_index(name = 'count'))
# %%
# null 존재?
print(all_list.isnull().values.any())
# null 없음
# %%
# 한글 정규표현식
all_list[:5]
# %%
all_list['뉴스'] = all_list['뉴스'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# 한글과 공백을 제외하고 모두 제거
all_list[:5]
# %%
# 특수문자로만 이루어진 문장들은 null 이 됐을 수 있다
print(all_list.isnull().values.any()) 
# null 없음
# %%
# SPLIT
from sklearn.model_selection import train_test_split
X = all_list['뉴스'].values
y = all_list['부도여부'].values
# %%
# training set / test set
train_data, test_data, train_label, test_label = train_test_split(X, y, random_state=42)
# %%
'''
3. 토큰화
토큰화 과정에서 불용어를 제거 
한국어의 조사, 접속사 등의 보편적인 불용어를 사용
'''
# %%
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
# %%
okt = Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔')
# %%
# Okt는  KoNLPy에서 제공하는 형태소 분석기
okt = Okt()
okt.morphs('와 이런 것도 영화라고 차라리 뮤직비디오를 만드는 게 나을 뻔', stem = True)
# %%
import timeit
start = timeit.default_timer()
okt = Okt()
X_train = []
y_train = []
i = 0
while i < len(train_data):
    sentence = train_data[i]
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)
    y_train.append(train_label[i])
    i += 10
stop = timeit.default_timer()
print('Time: ', stop - start, ', count ', len(X_train))  
# Time:  71.62022727200383 , count  22271
# %%
print(X_train[:3])
# %%
print(X_train[-3:])
# %%
start = timeit.default_timer()
X_test = []
y_test = []

i = 0
while i < len(test_data):
    sentence = test_data[i]
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)
    y_test.append(test_label[i])
    i += 25
stop = timeit.default_timer()
print('Time: ', stop - start, ', count ', len(X_test))  
# Time:  21.82575288099906 , count  3712
# %%
print(X_test[:3])
# %%
'''
4. 정수 인코딩
이제 기계가 텍스트를 숫자로 처리할 수 있도록 데이터에 정수 인코딩을 수행
훈련 데이터에 대해서 단어 집합(vocaburary)을 만들기
'''
# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
# %%
print(tokenizer.word_index) #빈도수 순서로 index
# %%
# print(tokenizer.texts_to_matrix(X_train[:5], mode = 'tfidf').round(2).mean(axis=1))
# %%
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)
# %%
# 전체 단어 개수 중 빈도수 2이하인 단어 개수는 제거.
# 0번 패딩 토큰과 1번 OOV 토큰을 고려하여 +2
vocab_size = total_cnt - rare_cnt + 2
print('단어 집합의 크기 :',vocab_size)
# %%
tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
# X_all = tokenizer.texts_to_sequences(X_all)
# %%
print(X_train[:5])
# print(X_test[:3])
# %%
# 각 샘플 내의 단어들은 각 단어에 대한 정수로 변환된 것을 확인할 수 있습니다. 
# 이제 단어의 개수는 19,417개로 제한되었으므로 0번 단어 ~ 19,416번 단어까지만 사용합니다. 
# (0번 단어는 패딩을 위한 토큰, 1번 단어는 OOV를 위한 토큰입니다.) 
# 이제 train_data에서 y_train과 y_test를 별도로 저장해줍니다.
# y_train = np.array(train_data['label'])
# y_test = np.array(test_data['label'])
# %%
'''
5. 빈 샘플(empty samples) 제거
'''
# %%
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]
# %%
# 빈 샘플들을 제거
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.delete(y_train, drop_train, axis=0)
print(len(X_train))
print(len(y_train))
# %%
'''
6. 패딩
이제 서로 다른 길이의 샘플들의 길이를 동일하게 맞춰주는 패딩 작업을 진행해보겠습니다. 
전체 데이터에서 가장 길이가 긴 리뷰와 전체 데이터의 길이 분포를 알아보겠습니다.
'''
# %%
print('리뷰의 최대 길이 :',max(len(l) for l in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))
plt.hist([len(s) for s in X_train], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()
# %%
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))
# %%
max_len = 17
below_threshold_len(max_len, X_train)
# 전체 샘플 중 길이가 17 이하인 샘플의 비율: 86.67782076754158
# %%
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)
# X_all = pad_sequences(X_all, maxlen = max_len)
# %%
print(X_train[:5])
# %%
'''
[2] LSTM으로 우량/불량 기업 감성 분류하기
'''
# %%
# 모델을 만들어봅시다. 우선 필요한 도구들을 가져옵니다.
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# %%
# 임베딩 벡터의 차원은 100으로 정했고, 분류를 위해서 LSTM을 사용합니다.
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
# %%
# 검증 데이터 손실(val_loss)이 증가하면, 과적합 징후므로 검증 데이터 손실이 4회 증가하면 학습을 조기 종료(Early Stopping)합니다. 
# 또한, ModelCheckpoint를 사용하여 검증 데이터의 정확도(val_acc)가 이전보다 좋아질 경우에만 모델을 저장합니다.

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
# %%
# 에포크는 총 15번을 수행하겠습니다. 또한 훈련 데이터 중 20%를 검증 데이터로 사용하면서 정확도를 확인합니다.

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
# %%
loaded_model = load_model('best_model.h5')
y_test = np.array(y_test)
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
# 테스트 정확도: 0.8595
# %%
'''
[3] 예측해보기
'''
# %%
def sentiment_predict(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩 
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  if(score > 0.5):
    print("{:.2f}% 확률로 부도 기업입니다.\n".format(score * 100))
    print("{:.2f} 스코어입니다.\n".format(score))
  else:
    print("{:.2f}% 확률로 건전 기업입니다.\n".format((1 - score) * 100))
    print("{:.2f} 스코어입니다.\n".format(score))
# %%
for i in range(100)[::10]:
    print(train_data[i], train_label[i])
    sentiment_predict(train_data[i])
    print("*********************************************")
# %%
for i in range(100)[1::10]:
    print(test_data[i], test_label[i])
    sentiment_predict(test_data[i])
    print("*********************************************")
# %%
# 성능 평가하기
def sentiment_predict_score(new_sentence):
  new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
  new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
  encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
  pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
  score = float(loaded_model.predict(pad_new)) # 예측
  return score

# %%
'''
결합모형에 사용할 설명변수 만들기
'''
start = timeit.default_timer()
all_list.reset_index(inplace = True)
y_pred = []
for i in range(len(all_list))[2::10]:
    y_tmp = sentiment_predict_score(all_list['뉴스'][i])
    y_pred.append(y_tmp)
stop = timeit.default_timer()
print('Time: ', stop - start)  
# Time:  1391.1455685050023 
# Time:  3180.4259664969977
# %%
x_list = all_list[2::10]
x_list['스코어'] = y_pred
x_list.to_csv('data_company_score_6m_mng_2배.csv', encoding='utf-8-sig')
# %%
