import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
# %matplotlib inline
tf.__version__

def f1(x):
    '''
    Funcao a ser aprendida
    '''
    return 5 + 10 * x

''' Perceptron
O perceptron é uma "rede neural" de um só neurônio. 

No nosso caso, temos a rede mais simples possível, com uma só entrada
e uma só saída, sem ativação.

Temos 100 dados que serão usados para treinar 300 épocas do percéptron.

Vamos utilizar o modelo percéptron para aprender uma simples regressão
linear, o objetivo é faze-lo aprender uma simples equação linear e
tambem se acostumar com a sintaxe e funcionamento do TensorFlow.
'''

# xs = np.linspace(0,10,100)  # gera 100 valores no intervalo [0 -> 10]
                              # computa o valor de f1 nestes 100 valores (lost=2e-5)
# xs = np.linspace(0,1,100)   # gera 100 valores no intervalo [0 -> 1]
                              # computa o valor de f1 nestes 100 valores (lost=25)
xs = np.linspace(0,1,500)     # gera 100 valores no intervalo [0 -> 1]
                              # computa o valor de f1 nestes 500 valores (lost=3e-3)
ys = f1(xs)
print(len(xs), "xs=", xs)
print(len(ys), "ys=", ys)

#Definindo, compilando e treinando nosso modelo
model = tf.keras.Sequential([
    keras.Input(shape=(1,)),
    keras.layers.Dense(units=1),
])

model.compile(optimizer="sgd", loss="mean_squared_error")
model.fit(xs,ys,epochs=300)

model.summary()
print("prediction: " + str(model.predict([17]))+ "\treal value: " + str(f1(17)))

# A função evaluate retorna o "custo" (loss) da avaliação,
# definido na compilação. Nesse caso, o valor reportado é o
# erro quadrático médio (MSE).
val = np.linspace(0,10,63)
model.evaluate(x=val, y=f1(val))

fig, ax = plt.subplots()
ax.plot( val, f1(val), '-b', val, model.predict( val ), 'or',
         markersize = 4, linewidth = 0.5 )
ax.grid( True )
plt.show()
