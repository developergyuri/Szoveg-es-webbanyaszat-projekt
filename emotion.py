#!/usr/bin/env python3
# coding: utf-8

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier;
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier  


### Adatok beolvasása, kisbetűre konvertálás
dataset = pd.read_csv('./emotion_final.csv')
dataset.text.apply(lambda x: [w.lower() for w in x])
dataset.emotion.apply(lambda x: [w.lower() for w in x])
print(dataset.head())

### Tábla infók
dataset.info()
print()

### Az adathalmaz statisztika kirajzolása
plt.figure(figsize=(16,9))
count = dataset.emotion.value_counts()
sns.barplot(x=count.index, y=count)
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=90);

text = dataset.text.values
label = dataset.emotion.values                    

### Szófelhő
word= ' '.join(text)
clude= WordCloud(width=1000, height=1000, random_state= 21,min_font_size=15,max_font_size=119).generate(word)
plt.figure(figsize = (15,15))
plt.imshow(clude,interpolation='bilinear')
plt.axis('off')

### Előfeldolgozás: Adathalmaz vektorizálása
vectorizer = CountVectorizer(stop_words='english',max_df=0.8, min_df=0.001)
DT = vectorizer.fit_transform(dataset.text)

vocabulary_dict = vectorizer.vocabulary_
vocabulary_list = vectorizer.get_feature_names()
vocabulary = np.asarray(vocabulary_list)
stopwords = vectorizer.stop_words_
n_words = DT.shape[1]

DT_dense = DT.toarray();

tfidfconverter = TfidfTransformer()
DT_dense = tfidfconverter.fit_transform(DT_dense).toarray()

#### Adathalmaz felosztása
X_train, X_test, y_train, y_test = train_test_split(
     DT_dense, label, test_size=0.3, shuffle = True, random_state=2021)

###############################################
### Random Forest osztályozás
clf_RF = RandomForestClassifier(n_estimators=100, random_state=2021)
clf_RF.fit(X_train, y_train)

## Train adatok
y_pred_RF = clf_RF.predict(X_train)
print("Random Forest Train Pontosság:", accuracy_score(y_train, y_pred_RF))

## Test adatok
y_pred_RF = clf_RF.predict(X_test)
print("Random Forest Test Pontosság:", accuracy_score(y_test, y_pred_RF))

## Eredmények
plot_confusion_matrix(clf_RF, X_test, y_test, display_labels = list(dict.fromkeys(label))); 

print("\n{stat}".format(stat = classification_report(y_test, y_pred_RF,
                                    target_names=list(dict.fromkeys(label)))));

###############################################
### Multinomial Naive Bayes osztályozás
clf_MNB = MultinomialNB(alpha=1)
clf_MNB.fit(X_train, y_train)

## Train adatok
y_pred_MNB = clf_MNB.predict(X_train)
print("Multinomial Naive Bayes Train Pontosság:", accuracy_score(y_train, y_pred_MNB))

## Test adatok
y_pred_MNB = clf_MNB.predict(X_test)
print("Multinomial Naive Bayes Test Pontosság:", accuracy_score(y_test, y_pred_MNB))

## Eredmények
plot_confusion_matrix(clf_MNB, X_test, y_test, display_labels = list(dict.fromkeys(label))); 

print("\n{stat}".format(stat = classification_report(y_test, y_pred_MNB,
                                    target_names=list(dict.fromkeys(label)))));

###############################################
### Stochastic Gradient Descent osztályozás
clf_SGD = SGDClassifier(loss="log", penalty='l1', alpha=0.0001, random_state=2021)
clf_SGD.fit(X_train,y_train)

## Train adatok
y_pred_SGD = clf_SGD.predict(X_train)
print("SGD Train Pontosság:", accuracy_score(y_train, y_pred_SGD))

## Test adatok
y_pred_SGD = clf_SGD.predict(X_test)
print("SGD Test Pontosság:", accuracy_score(y_test, y_pred_SGD))

## Eredmények
plot_confusion_matrix(clf_SGD, X_test, y_test, display_labels = list(dict.fromkeys(label))); 

print("\n{stat}".format(stat = classification_report(y_test, y_pred_SGD,
                                    target_names=list(dict.fromkeys(label)))));

