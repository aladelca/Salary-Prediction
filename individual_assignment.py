## Predicting salary

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from sklearn.naive_bayes import MultinomialNB
from utils.functions import *
import os
import warnings
import re
warnings.filterwarnings("ignore")

### Initial transformation


## Reading data
data = pd.read_csv('data/Train_rev1.csv')

## Sampling according to the requirement

df = data.sample(2500, random_state=123)

## Splitting data into train and test
x = df[['FullDescription']]
y = df['SalaryNormalized']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)

## Generating target variable using training variable

p75 = y_train.quantile(0.75)
y_train_cat = np.where(y_train > p75, 1, 0)
y_test_cat = np.where(y_test > p75, 1, 0)

## Analyzing target variable

sns.histplot(x = y_train)
plt.show()

### Data Preprocessing

x_train['cleaned_description'] = x_train['FullDescription'].apply(lambda x: clean_data(x))
x_test['cleaned_description'] = x_test['FullDescription'].apply(lambda x: clean_data(x))

## Tokenizing the data

regexp = RegexpTokenizer('\w+')
x_train['cleaned_description_token'] = x_train['cleaned_description'].apply(lambda x: regexp.tokenize(x))
x_test['cleaned_description_token'] = x_test['cleaned_description'].apply(lambda x: regexp.tokenize(x))

## Remove stop words

stopwords = nltk.corpus.stopwords.words("english")

## Add custom stop words

my_stopwords = ['https','www']
stopwords.extend(my_stopwords)

x_train['cleaned_description_token'] = x_train['cleaned_description_token'].apply(lambda x: [word for word in x if word.lower() not in stopwords])
x_test['cleaned_description_token'] = x_test['cleaned_description_token'].apply(lambda x: [word for word in x if word.lower() not in stopwords])

## Lemmatization
wordnet_lem = WordNetLemmatizer()
x_train['cleaned_description_token'] = x_train['cleaned_description_token'].apply(lambda x: [wordnet_lem.lemmatize(word) for word in x])
x_test['cleaned_description_token'] = x_test['cleaned_description_token'].apply(lambda x: [wordnet_lem.lemmatize(word) for word in x])


## Joining the words

x_train['final_description'] = x_train['cleaned_description_token'].apply(lambda x: ' '.join(x))
x_test['final_description'] = x_test['cleaned_description_token'].apply(lambda x: ' '.join(x))  

## Vectorizing the data

vect = CountVectorizer(stop_words='english',  ngram_range=(2,6))
x_train_vect = vect.fit_transform(x_train['final_description'])
x_test_vect = vect.transform(x_test['final_description'])

### Modeling

model_nb = MultinomialNB()
model_nb.fit(x_train_vect, y_train_cat)
preds = model_nb.predict(x_test_vect)
proba = model_nb.predict_proba(x_test_vect)
metrics, cm = get_metrics(preds, proba, y_test_cat, True)
print(metrics)

### Optimizing the model

model_nb = MultinomialNB()

min_range = [1,2,3,4,5,6,7,8,9,10]
max_range = [1,2,3,4,5,6,7,8,9,10]

list_auc = []
combinations = []
for i in max_range:
    for j in min_range:
        if j > i:
            continue
        else:
            auc = main_process(j, i, x_train, x_test, y_train_cat, y_test_cat, model_nb)
            print(auc)
            combination = (i,j)
            combinations.append(combination)
            list_auc.append(auc)

print('Best combination:',combinations[np.argmax(list_auc)])

## Best model
final_model = MultinomialNB()
x_train_esc, x_test_esc, _  = vectorizing(1,1,x_train, x_test)
final_model = training(final_model, x_train_esc, y_train_cat)
preds, proba = predict(final_model, x_test_esc)
metrics,_ = get_metrics(preds, proba, y_test_cat, True)
print(metrics)

plot_roc_auc(y_test_cat, proba)

precision, recall, threshold = plot_precision_recall(y_test_cat, proba)

## Optimizing threshold using Youden index
final_threshold = threshold[np.argmax(recall + precision-1)]

### Final model

final_model_w_optimized_threshold = MultinomialNB()
x_train_esc, x_test_esc, vect = vectorizing(1,1,x_train, x_test)
final_model_w_optimized_threshold = training(final_model_w_optimized_threshold, x_train_esc, y_train_cat)
preds, proba = predict(final_model, x_test_esc)

preds = np.where(proba[:,1] > final_threshold, 1, 0)
metrics,_ = get_metrics(preds, proba, y_test_cat, True)
print(metrics)

### Interpretation

prob_pos = sum(np.where(y_train_cat == 1 ,1,0))/len(y_train_cat)
prob_neg = sum(np.where(y_train_cat == 0 ,1,0))/len(y_train_cat)

df_nbf = pd.DataFrame()
df_nbf.index = vect.get_feature_names_out()
# Convert log probabilities to probabilities. 
df_nbf['pos'] = np.e**(final_model_w_optimized_threshold.feature_log_prob_[0, :])
df_nbf['neg'] = np.e**(final_model_w_optimized_threshold.feature_log_prob_[1, :])

 
df_nbf['odds_positive'] = (df_nbf['pos']/df_nbf['neg'])*(prob_pos /prob_neg)

df_nbf['odds_negative'] = (df_nbf['neg']/df_nbf['pos'])*(prob_neg/prob_pos )

odds_pos_top10 = df_nbf.sort_values('odds_positive',ascending=False)['odds_positive'][:10]
odds_neg_top10 = df_nbf.sort_values('odds_negative',ascending=False)['odds_negative'][:10]

print(odds_pos_top10)

print(odds_neg_top10)

### Next steps

#To increase the accuracy of the model, I can try it performing the following actions:

#1. Change the data cleaning steps. Now I only considered to keep numbers and letters, but I did not performed any cleaning about contractions (maybe identify them and transform them into the large version).
#2. Another step in the data cleaning could be identifying specific business jargon (like some certifications and licenses need such as CFA or I/II level) so they can be considered as input
#3. In the data preprocessing, I could try with stemming instead of lemmatization, to see the performance
#4. Also, in the data preprocessing step, I can identify more custom stop words to not including in the analysis
#5. In the vectorization part, I can try other vectorizers such as TfIdf vectorizer and Hashing vectorizer.
#6. In the vectorization part, I could have tried with more options in the ngrams to look for a better performance
