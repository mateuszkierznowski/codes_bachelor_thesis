import pickle
import gensim
from nltk import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from preprocesing.komentarze import Komentarze

with open('../objects/lematized_sentences', 'rb') as f:
    lematized = pickle.load(f)

all_wr = [word_tokenize(sen) for sen in lematized]
#load word2vec in order to transfer word to vektors
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(r'C:\Users\Mati\Desktop\w2vModel\nkjp+wiki-forms-all-100-cbow-hs_small.txt', binary=False, encoding='utf-8')
#vectorizing
def make_wektors(korpus):
    wektor_lst = []
    unfind_words = []
    all_w = korpus
    for sen in all_w:
        lst = []
        for i in range(20):
            if len(sen) > i:
                try:
                    lst.append(word2vec_model[sen[i]])
                except Exception as e:
                    unfind_words.append(sen[i])
                    lst.append(np.zeros(100, ))
            else:
                lst.append(np.zeros(100, ))
        wektor_lst.append(lst)
    return wektor_lst, unfind_words

wek, unfidn = make_wektors(all_wr)
# len(set(unfidn)) = 763

zzz = np.array(wek)
# np.save(r'objects/wektors', zzz)

###PART PREPARATION###
dicti = {'Groźby karalne': 1,
         'Pozostałe': 0,
         'Obraźliwe': 1,
         'Złośliwe': -1,
         'Krytyka': -1,
         'Ostra krytyka': -1}
k = Komentarze('../Komentarze').load()

#part words
df = k.get_dataframe()
df['klasa'] = df['klasa'].apply(lambda x: dicti[x])
short_wek_lst = [e for i, e in enumerate(wek) if df['klasa'][i] >= 0]

zzz = np.array(short_wek_lst)
#  np.save(r'objects/wektors_part', zzz)


