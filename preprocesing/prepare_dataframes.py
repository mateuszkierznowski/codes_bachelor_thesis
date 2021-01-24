from preprocesing.komentarze import Komentarze

k = Komentarze('../Komentarze').load()

df = k.get_dataframe()
df['klasa']

dicti = {'Groźby karalne': 1,
         'Pozostałe': 0,
         'Obraźliwe': 1,
         'Złośliwe': 1,
         'Krytyka': 0,
         'Ostra krytyka': 0}
df['klasa'] = df['klasa'].apply(lambda x: dicti[x])

df.to_csv(r'dataframes/full_csv')

###from nltk import word_tokenize

dicti = {'Groźby karalne': 1,
         'Pozostałe': 0,
         'Obraźliwe': 1,
         'Złośliwe': -1,
         'Krytyka': -1,
         'Ostra krytyka': -1}

df['klasa'] = df['klasa'].apply(lambda x: dicti[x])

df = df[df['klasa'] >= 0]

df.reset_index(inplace=True)

df = df.drop(columns=['level_0', 'index'])

df.to_csv(r'dataframes\binary_cls.csv', encoding='utf-8')

# res_wek is deklared in model_preparation
