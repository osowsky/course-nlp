'''
Nessa lista nós exploraremos o espaço vetorial gerado pelo algoritmo Word2Vec
e algumas de suas propriedades mais interessantes. Veremos como palavras
similares se organizam nesse espaço e as relações de palavras com seus
sinônimos e antônimos. Também veremos algumas analogias interessantes que o
algoritmo é capaz de fazer ao capturar um pouco do nosso uso da língua
portuguesa.
'''

import wget # dont install wget, instead install python3-wget
import os

# Docs do gensim
# https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.FastTextKeyedVectors.most_similar
from gensim.models import KeyedVectors

# Download the dataset.
CUR_DIR = os.path.dirname( os.path.abspath( __file__ ) ) + '/'
DATA_DIR = CUR_DIR + 'data/'
FILENAME = 'word2vec_200k.txt'
DATAFILE = DATA_DIR + FILENAME
DATA_URL = 'https://raw.githubusercontent.com/alan-barzilay/NLPortugues/master/Semana%2004/data/' + FILENAME

if ( not os.path.isfile( DATAFILE ) ):
    wget.download( DATA_URL, out = DATA_DIR )

# Load the  Word2vec method.
model = KeyedVectors.load_word2vec_format( DATAFILE )

# Most similar words (homônimas).
print( *model.most_similar( positive = 'manga' ), sep = '\n' )
print()
print( *model.most_similar( positive = 'morro' ), sep = '\n' )
print()

# Sinônimos e antônimos
print( model.distance( 'feio', 'belo' ) )
print( model.distance( 'feio', 'lindo' ) )
print( model.distance( 'lindo', 'belo' ) )

# Analogias.
print( *model.most_similar( positive = [ 'mulher', 'rei' ],
                            negative = ['homem'] ), sep = '\n' )
