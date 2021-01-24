# codes for bachelor's thesis
## Description
The main idea is to create a model with can easly detect toxic speech. The database is scraped from Wirtualna Polska and properly tagged. <br/>
The goal is to find the best combination of strings vectorizing and ML model. <br/>
Codes can be seperated into 2 categories:<br/>
  1) Preprocesing data
  2) Creation of a models
I divided data into full and partial in order to create 2 separated data base<br/>
# Preprocesing
In Komentarze.py are comments scrapped from Wirtualna Polska, and tagged within categories: Groźby karalne, obraźliwe, złośliwe, krytykam, ostra krytyka, pozostałe.<br/>
In Lematized.py the prepare data for lematization: delete stop_words and nubmers, correct mistakes, and finally lematize all comments.<br/>
In prepare_dataframes.py scripts create a dataframes (for full and partial)<br/>
In word_2_vec_preaparation.py scripts transfers comments into vektors creating word2vec<br/>
# Bow
In this folder scripts created ngrams bow vektors and put them into SVM, Random Forest, Gauss model, and gradient boost.<br/>
The best results are for unigram without long tails, (occurencess > 2), unfortunatelly bigrams, and trgigrams does not perform well,<br/>
Probably there are not enougth data to cover the problem.<br/>
# word2vec
In this folder scripts created word2vec vektors and put them into SVM, Random Forest, Gauss model, and gradient boost.<br/>
There is problem with a tremendous number of input (2000), however SVM can easily compile great results (another models doesnt perform well)<br/>
# doc2vec
In this folder scripts created doc2vec vektors and put them into SVM, Random Forest, Gauss model, and gradient boost.<br/>
This metod results better than word2vec (about 1-2%).<br/>
# Results
The outcome are diffrents, some models tend to have stabilize accurace % other fluctuate depends on models and type of string vectorizing.<br/>
However the best results are Random Forest with unigrams and deleted long tails (occurances > 3): 79,2%<br/>

