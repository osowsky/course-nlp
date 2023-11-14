import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wget # dont install wget, instead install python3-wget
import os

from sklearn.model_selection import train_test_split

print( tf.__version__ )

# Download the dataset.
CUR_DIR = os.path.dirname( os.path.abspath( __file__ ) ) + '/'
DATA_DIR = CUR_DIR + 'data/'
FILENAME = 'b2w-10k.csv'
DATAFILE = DATA_DIR + FILENAME
DATA_URL = 'https://raw.githubusercontent.com/alan-barzilay/NLPortugues/master/Semana%2003/data/' + FILENAME

if ( not os.path.isfile( DATAFILE ) ):
    wget.download( DATA_URL, out = DATA_DIR )

# Read dataset.
b2wCorpus = pd.read_csv( DATAFILE ) #, nrows = 1000 )
print( b2wCorpus.shape )
print( b2wCorpus.head() )

print( b2wCorpus["review_text"] )
print( b2wCorpus["reviewer_gender"].value_counts() )

# Get the columns 'review_text' and 'recommend_to_a_friend'
rev_txt = b2wCorpus['review_text']
print( rev_txt.shape )
rec_2_fr = []

# Convert 'recommend_to_a_friend' to a list of 0 and 1.
for rec in b2wCorpus['recommend_to_a_friend']:
  if ( rec == 'Yes' ):
    rec_2_fr.append( 1 )
  else:
    rec_2_fr.append( 0 )

# Vectorize the input text.
vectorize_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
  max_tokens=5000,
  standardize='lower_and_strip_punctuation',
  split='whitespace',
  ngrams=None,
  output_mode='int',
  output_sequence_length=None,
  pad_to_max_tokens=True )

vectorize_layer.adapt( rev_txt )
input = np.array( vectorize_layer( rev_txt ) )
print( f'input size: {input.shape}' )
output = np.array( rec_2_fr )
print( f'output size: {output.shape}' )

# Split the dataset to train and test the model.
X_train, X_test, Y_train, Y_test = train_test_split( input, output,
                                                     test_size = 0.33,
                                                     random_state = 42 )

# plt.hist([len(linha.split()) for linha in b2wCorpus["review_text"]])
# plt.show()

# Create the model.
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train.shape[ 1 ],) ),
    tf.keras.layers.Embedding(5000, 128, name="embedding"),
    tf.keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    tf.keras.layers.Conv1D(128, 7, padding='valid', activation='relu', strides=3),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# Train the NN and show a summary.
model.compile(optimizer="sgd", loss="mean_squared_error")
model.fit( x=X_train, y=Y_train, epochs=50 )
model.summary()

# Evaluate the NN.
model.evaluate( x=X_test, y=Y_test )




