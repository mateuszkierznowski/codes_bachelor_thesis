import spacy
from preprocesing.komentarze import Komentarze
import pickle
import re
from nltk.tokenize import word_tokenize
from autocorrect import Speller

k = Komentarze('C:\\Users\\Mati\\Desktop\\Praca Magr\\Praca Licencjacka\\praca_licencjacka_kody/Komentarze').load()

df = k.get_dataframe()
df['klasa']

dicti = {'Groźby karalne': 1,
         'Pozostałe': 0,
         'Obraźliwe': 1,
         'Złośliwe': 1,
         'Krytyka': 0,
         'Ostra krytyka': 0}
df['klasa'] = df['klasa'].apply(lambda x: dicti[x])

kom_lst = df['komentarz'].tolist()


#creating all with small letter
all_sen = [sen.lower() for sen in kom_lst]

# Ged rid of special mark
def delete_s_mark(word):
    return re.sub(r'[^\w]', " ", word)[:-1]


all_sen = [delete_s_mark(kom) for kom in all_sen]

#delete stopwords
with open('objects/stopwords.txt', 'r', encoding='utf-8') as f:
    stop_words = [re.sub(r'\n', '', line) for line in f]

def delete_stop_word(line):
    sentence = [token for token in word_tokenize(line) if not token in stop_words]
    return " ".join(sentence)

all_sen = [delete_stop_word(line) for line in all_sen]

#delete numbers
def delete_numbers(sen):
    sentence = [word for word in word_tokenize(sen) if not word.isdigit()]
    return " ".join(sentence)

all_sen = [delete_numbers(sen) for sen in all_sen]

#delete alone letters
def delete_numbers(sen):
    sentence = [word for word in word_tokenize(sen) if len(word) > 1]
    return " ".join(sentence)

all_sen = [delete_numbers(sen) for sen in all_sen]

#correct mistakes
spell = Speller(lang='pl')
# lem = [spell(word) for sen in all_w for word in sen]

def correct_mistakes(sen):
    words = [spell(word) for word in word_tokenize(sen)]
    return " ".join(words)

all_sen =  [correct_mistakes(sen) for sen in all_sen]

# all_wr = []
# for sen in all_sen:
#     lst = []
#     for word in sen:
#         lst.append(spell(word))
#     all_wr.append(lst)


#crate spacy lematizaer
nlp = spacy.load('pl_core_news_sm')


def lematize(statement):
    stat = nlp(statement)
    lst = [token.lemma_ for token in stat]
    return " ".join(lst)

#lematize
lst_lem = [lematize(stat) for stat in all_sen]

with open('../objects/lematized_sentences', 'wb') as f:
    pickle.dump(lst_lem, f)

### Lematized sentences part
dicti = {'Groźby karalne': 1,
         'Pozostałe': 0,
         'Obraźliwe': 1,
         'Złośliwe': -1,
         'Krytyka': -1,
         'Ostra krytyka': -1}
k = Komentarze('../Komentarze').load()
df = k.get_dataframe()
df['klasa'] = df['klasa'].apply(lambda x: dicti[x])

with open('../objects/lematized_sentences', 'rb') as f:
    lematized = pickle.load(f)

short_lem_lst = [element for number, element in enumerate(lematized) if df['klasa'][number] >= 0]

with open ('../objects/lematized_part_sentences', 'wb') as f:
    pickle.dump(short_lem_lst, f)
    f.close()