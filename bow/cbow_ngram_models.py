from sklearn.feature_extraction.text import CountVectorizer
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

sv = SVC()
RFC = RandomForestClassifier()
GaussianN = GaussianNB()
KNC = KNeighborsClassifier(n_neighbors=7)
xgboost = XGBClassifier()
gradientboost = GradientBoostingClassifier()

with open('objects/lematized_part_sentences', 'rb') as f:
    lematized_part_sentences = pickle.load(f)
df_partial = pd.read_csv('dataframes/binary_cls.csv', index_col=[0])

with open('objects/lematized_sentences', 'rb') as f:
    lematized_sentences = pickle.load(f)

df_full = pd.read_csv('dataframes/full_csv', index_col=[0])
def net(ngram_min, ngram_max, min_diff, input_data, df):
    vectorizer = CountVectorizer(ngram_range=(ngram_min, ngram_max), min_df=min_diff)
    n_gram_matrix = vectorizer.fit_transform(input_data).toarray()
    X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, df['klasa'], test_size=0.20)

    sv.fit(X_train, y_train)
    print('Precyzja dla modelu SVM: ' + str(sv.score(X_test, y_test)))

    RFC.fit(X_train, y_train)
    print('Precyzja dla modelu lasów losowych: ' + str(RFC.score(X_test, y_test)))

    GaussianN.fit(X_train, y_train)
    print('Precyzja dla modelu BayesG: ' + str(GaussianN.score(X_test, y_test)))

    KNC.fit(X_train, y_train)
    print('Precyzja dla modelu KNN: ' + str(KNC.score(X_test, y_test)))

    gradientboost.fit(X_train, y_train)
    print('Precyzja dla modelu GB: ' + str(gradientboost.score(X_test, y_test)))

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

for i in range(3,7):
    net(1, 1, i,lematized_part_sentences, df_partial)

for i in range(2,7):
    net(1, 1, i,lematized_sentences, df_full)

vectorizer = CountVectorizer(ngram_range=(2, 2), min_df=1)
n_gram_matrix = vectorizer.fit_transform(lematized_part_sentences).toarray()
X_train, X_test, y_train, y_test = train_test_split(n_gram_matrix, df_partial['klasa'], test_size=0.20)

sv.fit(X_train, y_train)
print('Precyzja dla modelu SVM: ' + str(sv.score(X_test, y_test)))

RFC.fit(X_train, y_train)
print('Precyzja dla modelu lasów losowych: ' + str(RFC.score(X_test, y_test)))

GaussianN.fit(X_train, y_train)
print('Precyzja dla modelu BayesG: ' + str(GaussianN.score(X_test, y_test)))

KNC.fit(X_train, y_train)
print('Precyzja dla modelu KNN: ' + str(KNC.score(X_test, y_test)))

gradientboost.fit(X_train, y_train)
print('Precyzja dla modelu GB: ' + str(gradientboost.score(X_test, y_test)))