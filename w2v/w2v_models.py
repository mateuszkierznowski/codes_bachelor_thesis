import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from varname import nameof


sv = SVC()
RFC = RandomForestClassifier()
GaussianN = GaussianNB()
KNC = KNeighborsClassifier(n_neighbors=7)
xgboost = XGBClassifier()
gradientboost = GradientBoostingClassifier()


df = pd.read_csv(r'dataframes/full_csv', index_col=[0])


# with open(r'objects/wektor_lst', 'rb') as f:
#     res_wek = np.load(f)
res_wek = np.load(r'objects/wektors.npy', allow_pickle=True)
res_wek = [wek[0:20] for wek in res_wek]
zzz = np.stack(res_wek)
res_wek = zzz.reshape([7023,2000])

scoring = ['precision', 'recall', 'f1', 'accuracy']

sv_score_array = cross_validate(sv, res_wek, df['klasa'], cv=5, scoring=scoring)
rfc = cross_validate(RFC, res_wek, df['klasa'], cv=5, scoring=scoring)
MNB_score_array = cross_validate(GaussianN, res_wek, df['klasa'], cv=5, scoring=scoring)
KNC_score_array = cross_validate(KNC, res_wek, df['klasa'], cv=5, scoring=scoring)

measure_df = pd.DataFrame(columns=['mode', 'precision', 'recall', 'f1', 'accuracy'])
dict_lst = [sv_score_array, rfc, MNB_score_array, KNC_score_array]

for dict in dict_lst:
    measure_df.loc[len(measure_df)] = [nameof(dict),
                         np.mean(dict['test_precision']),
                         np.mean(dict['test_recall']), np.mean(dict['test_f1']),
                         np.mean(dict['test_accuracy'])]

measure_df.to_csv('w2v/models_full_results.csv')

###Partial####
res_wek = np.load(r'objects/wektors_part.npy', allow_pickle=True)
res_wek = [wek[0:20] for wek in res_wek]
zzz = np.stack(res_wek)
res_wek = zzz.reshape([4269,2000])
df = pd.read_csv('dataframes/binary_cls.csv')
X_train, X_test, y_train, y_test = train_test_split(res_wek, df['klasa'], test_size=0.20)

sv.fit(X_train, y_train)
print(sv.score(X_test, y_test))
#acc 71,66%
RFC.fit(X_train, y_train)
print(RFC.score(X_test, y_test))
#acc 69,32%
GaussianN.fit(X_train, y_train)
print(GaussianN.score(X_test, y_test))
#acc 56,20%

KNC.fit(X_train, y_train)
print(KNC.score(X_test, y_test))
#acc 59,01%
gradientboost.fit(X_train, y_train)
print(gradientboost.score(X_test, y_test))
# ac 70,49%
xgboost.fit(X_train, y_train)
print(xgboost.score(X_test, y_test))
#acc = 73,18%

sv_score_array = cross_validate(sv, res_wek, df['klasa'], cv=5, scoring=scoring)
rfc = cross_validate(RFC, res_wek, df['klasa'], cv=5, scoring=scoring)
MNB_score_array = cross_validate(GaussianN, res_wek, df['klasa'], cv=5, scoring=scoring)
KNC_score_array = cross_validate(KNC, res_wek, df['klasa'], cv=5, scoring=scoring)

measure_df = pd.DataFrame(columns=['mode', 'precision', 'recall', 'f1', 'accuracy'])
dict_lst = [sv_score_array, rfc, MNB_score_array, KNC_score_array]

for dict in dict_lst:
    measure_df.loc[len(measure_df)] = [nameof(dict),
                                       np.mean(dict['test_precision']),
                                       np.mean(dict['test_recall']), np.mean(dict['test_f1']),
                                       np.mean(dict['test_accuracy'])]

measure_df.to_csv('w2v/models_partial_results.csv')