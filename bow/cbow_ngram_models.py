from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import numpy as np
from sklearn.model_selection import cross_validate
from varname import nameof


sv = SVC()
RFC = RandomForestClassifier()
MultinomiaNB = MultinomialNB()
KNC = KNeighborsClassifier(n_neighbors=7)


with open('objects/lematized_part_sentences', 'rb') as f:
    lematized_part_sentences = pickle.load(f)
df_partial = pd.read_csv('dataframes/binary_cls.csv', index_col=[0])

with open('objects/lematized_sentences', 'rb') as f:
    lematized_sentences = pickle.load(f)

df_full = pd.read_csv('dataframes/full_csv', index_col=[0])

def net(X_train, X_test, y_train, y_test):
    sv.fit(X_train, y_train)
    print('Precyzja dla modelu SVM: ' + str(sv.score(X_test, y_test)))

    RFC.fit(X_train, y_train)
    print('Precyzja dla modelu lasów losowych: ' + str(RFC.score(X_test, y_test)))

    MultinomiaNB.fit(X_train, y_train)
    print('Precyzja dla modelu BayesG: ' + str(MultinomiaNB.score(X_test, y_test)))

    KNC.fit(X_train, y_train)
    print('Precyzja dla modelu KNN: ' + str(KNC.score(X_test, y_test)))

    #gradientboost.fit(X_train, y_train)
    #print('Precyzja dla modelu GB: ' + str(gradientboost.score(X_test, y_test)))

#parital

net(1, 1, 2, lematized_part_sentences, df_partial) # unigram with more than 1 occuring
# Precyzja dla modelu SVM: 0.7517564402810304
# Precyzja dla modelu lasów losowych: 0.7740046838407494
# Precyzja dla modelu Gaussa: 0.7318501170960188
# Precyzja dla modelu KNN: 0.7318501170960188
# Precyzja dla modelu GB: 0.7107728337236534
net(2, 2, 2, lematized_part_sentences, df_partial)# bigrams
net(3, 3, 2, lematized_part_sentences, df_partial)# trigrams


# full
net(1, 1, 2, lematized_sentences, df_full)
# Precyzja dla modelu SVM: 0.7153024911032029
# Precyzja dla modelu lasów losowych: 0.7053380782918149
# Precyzja dla modelu Gaussa: 0.6256227758007118
# Precyzja dla modelu KNN: 0.5587188612099644
# Precyzja dla modelu GB: 0.6284697508896797

net(2, 2, 2, lematized_sentences, df_full)
# Precyzja dla modelu SVM: 0.6227758007117438
# Precyzja dla modelu lasów losowych: 0.6170818505338078
# Precyzja dla modelu Gaussa: 0.606405693950178
# Precyzja dla modelu KNN: 0.5708185053380783
# Precyzja dla modelu GB: 0.5530249110320284
net(3, 3, 2, lematized_sentences, df_full)

for i in range(1, 7):
    print("{} long tails iteration".format(i))
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=i)
    n_gram_matrix = vectorizer.fit_transform(lematized_part_sentences).toarray()
    X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, df_partial['klasa'], test_size=0.20)
    net(X_train, X_test, y_train, y_test)

for i in range(5, 6):
    print("{} long tails iteration".format(i))
    vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=i)
    n_gram_matrix = vectorizer.fit_transform(lematized_sentences).toarray()
    X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, df_full['klasa'], test_size=0.20)
    net(X_train, X_test, y_train, y_test)

sv = SVC()
RFC = RandomForestClassifier(n_estimators=100)
MultinomiaNB = MultinomialNB()
KNC = KNeighborsClassifier(n_neighbors=7)


vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
n_gram_matrix = vectorizer.fit_transform(lematized_part_sentences).toarray()
X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, df_partial['klasa'], test_size=0.20)

RFC.fit(X_train, y_train)
print('Precyzja dla modelu lasów losowych: ' + str(RFC.score(X_test, y_test)))

random.seed(23145)

for i in range(100, 500, 50):
    np.random.seed(42)
    RFC = RandomForestClassifier(n_estimators=1500)
    RFC.fit(X_train, y_train)
    print('Precyzja dla modelu lasów losowych {}: '.format(str(i)) + str(RFC.score(X_test, y_test)))


scoring = ['precision', 'recall', 'f1', 'accuracy']
###PARTIAL KFOLD WALIDATION####
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
n_gram_matrix = vectorizer.fit_transform(lematized_sentences).toarray()

sv_score_array = cross_validate(sv, n_gram_matrix, df_full['klasa'], cv=5, scoring=scoring)
rfc = cross_validate(RFC, n_gram_matrix, df_full['klasa'], cv=5, scoring=scoring)
MNB_score_array = cross_validate(MultinomiaNB, n_gram_matrix, df_full['klasa'], cv=5, scoring=scoring)
KNC_score_array = cross_validate(KNC, n_gram_matrix, df_full['klasa'], cv=5, scoring=scoring)

measure_df = pd.DataFrame(columns=['mode', 'precision', 'recall', 'f1', 'accuracy'])
dict_lst = [sv_score_array, rfc, MNB_score_array, KNC_score_array]

for dict in dict_lst:
    measure_df.loc[len(measure_df)] = [nameof(dict),
                                       np.mean(dict['test_precision']),
                                       np.mean(dict['test_recall']), np.mean(dict['test_f1']),
                                       np.mean(dict['test_accuracy'])]

measure_df.to_csv('bow/models_full_results.csv')

###FULL KFOLD WALIDATION####
vectorizer = CountVectorizer(ngram_range=(1, 1), min_df=2)
n_gram_matrix = vectorizer.fit_transform(lematized_sentences).toarray()




