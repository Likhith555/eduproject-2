import numpy as np
import pandas as pd

True_news = pd.read_csv(r"C:\Users\Likhith\OneDrive\Desktop\EDU VERSITY\True.csv")
Fake_news = pd.read_csv(r"C:\Users\Likhith\OneDrive\Desktop\EDU VERSITY\Fake.csv")
print(True_news)
print(Fake_news)

True_news['label'] = 0
Fake_news['label'] = 1
print(True_news,Fake_news)

dataset1 = True_news[['text','label']]
dataset2 = Fake_news[['text','label']]
print(dataset1,dataset2)

dataset = pd.concat([dataset1, dataset2])
print(dataset)
dataset.shape
dataset.isnull().sum()
dataset['label'].value_counts()
dataset = dataset.sample(frac =1)
print(dataset)

# NLP

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ps = WordNetLemmatizer()
stopwords = stopwords.words('english')


def clean_row(row):
    row = row.lower()
    row = re.sub('[^a-zA-Z]',' ',row)
    token = row.split()

    news = [ps.lemmatize(word) for word in token if not word in stopwords]

    cleanned_news = ' '.join(news)

    return cleanned_news
print(ps)

dataset['text'] = dataset['text'].apply(lambda x : clean_row(x))
print(dataset['text'])

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 50000, lowercase = False, ngram_range=(1, 2))
x = dataset.iloc[:35000, 0]
y = dataset.iloc[:35000, 1]
print(x, y)

from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = train_test_split(x, y, test_size = 0.2, random_state = 0)
vec_train_data = vectorizer.fit_transform(train_data)
vec_train_data = vec_train_data.toarray()
type(vec_train_data)

vec_test_data = vectorizer.fit_transform(test_data)
vec_test_data = vec_test_data.toarray()
vec_train_data.shape, vec_test_data.shape
print(vec_train_data)

train_data = pd.DataFrame(vec_train_data, columns = vectorizer.get_feature_names_out())
testing_data = pd.DataFrame(vec_test_data, columns = vectorizer.get_feature_names_out())
print(train_data)

# model

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_data, train_label)
y_pred = clf.predict(testing_data)
print(test_label, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_label, y_pred))

y_pred_train = clf.predict(train_data)
print(accuracy_score(train_label, y_pred_train))

txt = ('The following statements were posted to the verified Twitter accounts of U.S. President Donald Trump, @realDonaldTrump and @POTUS.  The opinions expressed are his own. Reuters has not edited the statements or confirmed their accuracy.  @realDonaldTrump : - “On 1/20 - the day Trump was inaugurated - an estimated 35,000 ISIS fighters held approx 17,500 square miles of territory in both Iraq and Syria. As of 12/21, the U.S. military estimates the remaining 1,000 or so fighters occupy roughly 1,900 square miles...” via @jamiejmcintyre  [1749 EST] - Just left West Palm Beach Fire & Rescue #2. Met with great men and women as representatives of those who do so much for all of us. Firefighters, paramedics, first responders - what amazing people they are! [1811 EST] - “On 1/20 - the day Trump was inaugurated - an estimated 35,000 ISIS fighters held approx 17,500 square miles of territory in both Iraq and Syria. As of 12/21, the U.S. military est the remaining 1,000 or so fighters occupy roughly 1,900 square miles..” @jamiejmcintyre @dcexaminer [2109 EST] - "Arrests of MS-13 Members, Associates Up 83% Under Trump" bit.ly/2liRH3b [2146 EST] -- Source link: (bit.ly/2jBh4LU) (bit.ly/2jpEXYR) ')

news = clean_row(txt)
print(news)

pred = clf.predict(vectorizer.transform([news]).toarray())
print(pred)

txt = input("Enter News")

news = clean_row(str(txt))

pred = clf.predict(vectorizer.transform([news]).toarray())

if pred == 0:
    print('News is correct')
else:
    print('News is fake')




