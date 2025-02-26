import pandas as pd
import string, re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


#memasukkan dataset
df_tweet = pd.read_csv("D:\SKRIPSI\Dataset\datasetpilkada.csv")
df_tweet=df_tweet.drop(columns=['Pasangan Calon'])
df_tweet=df_tweet.drop(columns=['Id'])

label=[]
for index, row in df_tweet.iterrows():
    if row["Sentiment"]=='negative':
        label.append(1)
    else:
        label.append(0)
df_tweet["label"]=label
df_tweet=df_tweet.drop(columns=['Sentiment'])
banyak = df_tweet['label'].value_counts()

df_process=df_tweet.copy()
s_1=df_process[df_process["label"]==1].sample(300,replace=True)
s_2=df_process[df_process["label"]==0].sample(300,replace=True)
df_process=pd.concat([s_1,s_2])

#preprocessing data
def cleansing(data):
    data=data.lower()
    remove=string.punctuation
    translator=str.maketrans(remove, ' '*len(remove))
    data=data.translate(translator)
    data=data.encode('ascii', 'ignore').decode('utf-8')
    data=re.sub(r'[^\x00-\x7f]',r'',data)

    factory=StopWordRemoverFactory()
    stopword=factory.create_stop_word_remover()
    data=stopword.remove(data)

    factory=StemmerFactory()
    stemmer=factory.create_stemmer()
    data=stemmer.stem(data)

    return data

cleaning=[]
for index, row in df_process.iterrows():
    cleaning.append(cleansing(row["Text Tweet"]))
df_process["Text Tweet"]=cleaning

X_train,X_test,y_train,y_test = train_test_split(df_process["Text Tweet"],df_process["label"],
                                                 test_size=0.16, stratify=df_process["label"],random_state=30)
vectorizer=TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors._ball_tree import BallTree

import math
#def euclidean(x,y):
    #return math.sqrt(sum((x - y)**2))
#def manhattan(x,y):
    #return sum(abs(x - y))
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)
hasil=classifier.predict(X_test)

from sklearn.metrics import f1_score,recall_score,precision_score,confusion_matrix,accuracy_score
#akurasi dll
print("Akurasi yang Didapat adalah :")
print(accuracy_score(y_test,hasil))
print("f1 score yang Didapat adalah :")
print(f1_score(y_test,hasil))
print("Recall yang Didapat adalah :")
print(recall_score(y_test,hasil))
print("Precision yang Didapat adalah :")
print(precision_score(y_test,hasil))

#Confusion Matriks
tn,fp,fn,tp=confusion_matrix(y_test,hasil).ravel()
print(tn,fp,fn,tp)























#data testing
#HS= "awas banyak yang mendadak baik, metamorfosis dari kecebong brubah jd anjing penjilat"
#non_HS="tetap semangat jangan putus asa pak ahy #Pilkadadki"
#Pernyataan=input("Masukkan Pernyataan :")
#cek1=classifier.predict(preprocess_data(Pernyataan))
#if cek1==[1]:
    #print("Hate Speech")
#else:
    #print("NON Hate Speech")
#cek2=classifier.predict(preprocess_data(non_HS))
#print("PENGUJIAN PROGRAM")
#print("==================================================================================")
#print("Ket")
#print("[1] = Hate Speech(HS)   [0] = Non Hate Speech(non_HS)")
#print("Pernyataan 1 (HS) = awas banyak mendadak baik, metamorfosis dari kecebong brubah jd anjing penjilat")
#print("Pernyataan 2 (non_HS) = tetap semangat jangan putus asa pak ahy")
#print("==================================================================================")
#print("Hasil")
#print("Pernyataan 1 = ")
#print(cek1)
#print("Pernyataan 2 = ")
#print(cek2)


