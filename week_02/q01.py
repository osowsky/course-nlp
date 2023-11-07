import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
# %matplotlib inline
tf.__version__

# def f1(x):
#     '''
#     Funcao a ser aprendida
#     '''
#     return 5 + 10 * x

def f2(x):
    '''
    Funcao não linear a ser aprendida
    '''
    return (x**2 + x*3 + 4)/200

''' Perceptron
O perceptron é uma "rede neural" de um só neurônio. 

No nosso caso, temos a rede mais simples possível, com uma só entrada
e uma só saída, sem ativação.

Temos 100 dados que serão usados para treinar 300 épocas do percéptron.

Vamos utilizar o modelo percéptron para aprender uma simples regressão
linear, o objetivo é faze-lo aprender uma simples equação linear e
tambem se acostumar com a sintaxe e funcionamento do TensorFlow.
'''

xs = np.linspace(0,10,100)  # gera 100 valores no intervalo [0 -> 10]
ys = f2(xs)                 # computa o valor de f1 nestes 100 valores
print(len(xs), "xs=", xs)
print(len(ys), "ys=", ys)

#Definindo, compilando e treinando nosso modelo
model = tf.keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=4,activation='tanh'),
    keras.layers.Dense(units=4,activation='tanh'),
    keras.layers.Dense(units=4,activation='tanh'),
    keras.layers.Dense(units=1),
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(xs,ys,epochs=400)

model.summary()
print("prediction: " + str(model.predict([17]))+ "\treal value: " + str(f2(17)))

# A função evaluate retorna o "custo" (loss) da avaliação,
# definido na compilação. Nesse caso, o valor reportado é o
# erro quadrático médio (MSE).
val = np.linspace(0,10,63)
model.evaluate(x=val, y=f2(val))

fig, ax = plt.subplots()
ax.plot( val, f2(val), '-b', val, model.predict( val ), 'or',
         markersize = 4, linewidth = 0.5 )
ax.grid( True )
plt.show()
