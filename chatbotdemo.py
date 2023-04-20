
import pandas as pd
import os
import random
import string
import gensim
from gensim import corpora, models, similarities
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('wordnet')

import warnings
warnings.simplefilter('ignore')

# Data Preparation and Preprocessing

print("Loading Data Sources")
data = pd.read_csv("dataM.csv")
data.head(50)

temp_data =  pd.read_csv("dataM.csv")
question_data = temp_data['message']

#Create Stop Word
newstopwords = set(stopwords.words('english'))

#define Wordnet Lemmatizer 
WNlemma = nltk.WordNetLemmatizer()

#Create Preprocessing Function
#This section defines a function pre_process() which performs text preprocessing on the input text. 
#The function tokenizes the input text into individual words, 
#lemmatizes each word using WordNet's lemmatizer, removes punctuation and stopwords,
#and returns the preprocessed list of tokens.
def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens= [ t for t in tokens if t not in string.punctuation ]
    tokens=[word for word in tokens if word.lower() not in newstopwords]
    # bigr = nltk.bigrams(tokens[:10])
    # trigr = nltk.trigrams(tokens[:10])
    return(tokens)

import random
#This section defines a function greeting() which checks if the input sentence contains any greetings 
#from the GREETING_INPUTS tuple. If a greeting is found, a random response is selected from the 
#GREETING_RESPONSES list and returned.
GREETING_INPUTS = ("hello", "hi", "greetings", "hello i need help", "good day", "hey", "i need help", "greetings")
GREETING_RESPONSES = ["Hello! How can I assist you today?", "Hello, How can I help you?", "Hello", "I am glad! You are talking to me.", "hello! If you have any more questions, feel free to ask!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
#This section preprocesses the message column of the data by applying the 
#pre_process() function to each element of the column using the apply() method of Pandas dataframes. 
#The original message column is replaced by the preprocessed version. question_data is created to store
#Preprocess Question Column
question_data = data['message']
data['message'] = data['message'].apply(pre_process)
print(question_data)

#Define Questions
question = data['message']
dictionary = corpora.Dictionary(question)
corpus = [dictionary.doc2bow(a) for a in question]
tfidf = models.TfidfModel(corpus)  
corpus_tfidf = tfidf[corpus]

#applying LSI model
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=650) 
corpus_lsi = lsi[corpus_tfidf]
index = similarities.MatrixSimilarity(corpus_lsi)

# ChatBot Function Definition

def Talk_To_bot(test_set_sentence):        
    # Tokenisation of user input
    tokens = pre_process(test_set_sentence)
    texts = " ".join(tokens)

    # Find and Sort Similarity 
    vec_bow = dictionary.doc2bow(texts.lower().split())
    vec_tfidf = tfidf[vec_bow]
    vec_lsi = lsi[vec_tfidf]

    #If not in the topic trained.
    if not (vec_lsi):
        
        not_understood = "Apology, I do not understand. Can you rephrase?"
        return not_understood, 999
    
    else: 
        # sort similarity
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        
        index_s =[]
        score_s = []
        for i in range(len(sims)):
            x = sims[i][1]
            # If similarity is less than 0.2 ask user to rephrase.
            if x <=0.8: # Threshold B
                index_s.append(str(sims[i][0]))
                score_s.append(str(sims[i][1]))
                reply_indexes = pd.DataFrame({'index': index_s,'score': score_s})

                r_index = int(reply_indexes['index'].loc[0])
                r_score = float(reply_indexes['score'].loc[0])

                not_understood = "Apology, I do not understand. Can you rephrase?"
                return not_understood, 999
            else: 
                index_s.append(str(sims[i][0]))
                score_s.append(str(sims[i][1]))
                reply_indexes = pd.DataFrame({'index': index_s,'score': score_s})
        

            #Find Top Questions and Score  
            r_index = int(reply_indexes['index'].loc[0])
            r_score = float(reply_indexes['score'].loc[0])
            reply = str(data.iloc[:,1][r_index])
        
            return reply, r_score

def lsa(input_sentence): 
    if input_sentence.lower() != 'bye':
        if greeting(input_sentence.lower()) is not None:
            return greeting(input_sentence.lower())
        else:
            reply, score = Talk_To_bot(str(input_sentence))
            return reply


import streamlit as st


st.title("My Chatbot App")
# Get user input
user_input = st.text_input("Enter your message here:")
# Get chatbot response
if user_input:
    response = lsa(user_input)
    st.write(response)
#Next, the code defines a list of possible user greetings (GREETING_INPUTS) and a 
#list of corresponding bot responses (GREETING_RESPONSES). 
#It also defines a greeting() function that takes a sentence as input 
#and returns a random greeting response if any of the words in the sentence 
#match the words in GREETING_INPUTS.

#The code then preprocesses the question data in the message column of the dataset using the 
#pre_process() function. This function tokenizes the text, lemmatizes the tokens, 
#removes punctuation and stop words. The resulting preprocessed questions are then used
#to create a dictionary, a corpus, and a tf-idf model.

#Next, the code applies Latent Semantic Analysis (LSA) using the models.LsiModel() function to 
#the tf-idf corpus. This creates a new corpus (corpus_lsi) that 
#represents each question in terms of a smaller number of topics. Finally, 
#a similarity index (index) is created using the similarities.MatrixSimilarity() function.

#La fonction Talk_To_bot() est définie pour accepter une phrase saisie par l'utilisateur, la prétraiter 
#en utilisant la fonction pre_process() , puis recherchez la question la plus similaire dans l'ensemble 
#de données.Il le fait en transformant d'abord la phrase d'entrée en un vecteur de sac de mots (vec_bow)
#en utilisant le dictionnaire, puis en transformant ce vecteur en un vecteur tf-idf (vec_tfidf) 
#en utilisant le modèle tf-idf,et enfin transformer ce vecteur en un vecteur LSA (vec_lsi) 
#en utilisant le modèle LSA.

#Si le vecteur LSA est vide (c'est-à-dire que la phrase d'entrée n'est pas dans le sujet des questions formées),
#la fonction renvoie un message d'excuse. Sinon, la fonction calcule
#la similarité entre le vecteur LSA et tous les vecteurs LSA du jeu de données en utilisant l'indice de 
#similarité,
#trie les résultats par ordre décroissant de similarité et renvoie la question la plus similaire
#et son score de similarité.

#La fonction lsa() est définie pour accepter une phrase saisie par l'utilisateur, vérifiez s'il s'agit d'un message d'accueil à l'aide de la
#greeting() , puis transmettez la phrase d'entrée à Talk_To_bot() pour obtenir une réponse.
#Si l'utilisateur saisit "bye", la fonction s'arrête.

#Enfin, l'application Streamlit est définie pour accepter la saisie de l'utilisateur à l'aide d'un widget de saisie de texte, et
#puis passez l'entrée à la fonction lsa() pour obtenir une réponse. La réponse est affichée à l'aide d'un widget de texte.