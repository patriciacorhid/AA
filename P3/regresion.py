# -*- coding: utf-8 -*-
import numpy as np
import math as m
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.decomposition import PCA
import sklearn.preprocessing as proc
from sklearn.model_selection import train_test_split
import pandas as pd

# Fijamos la semilla
np.random.seed(1)

#------------------PROCESAMIENTO DE DATOS--------------------------

#Leemos los datos del fichero
def readData(file):
	# Leemos los ficheros
        data = pd.read_csv(file, header=None, na_values = '?')

        x = data.iloc[:, :-1] #Datos socio-economicos
        y = data.iloc[:, -1]  #Número de crimenes
               
        return x, y

#------------------PRACTICA 3--------------------------

#Leemos los datos del fichero
x, y = readData('datos/communities.data')

#Establezco la longitud del conjunto de los conjuntos de training y test (80%, 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#PREPROCESADO DE DATOS
#Quitamos las 5 caractericticas no predicativas:
x_train = x_train.drop([0,1,2,3,4], axis = 1)
x_test = x_test.drop([0,1,2,3,4], axis = 1)

#print(x_train)
#Quitamos características con el 99% de valores incógnita
l = int(99*len(x_train.index)/100)
x_train = x_train.dropna(thresh = l, axis = 1)
#print(x_train.columns)
x_test = x_test.reindex(columns = x_train.columns)
#print(x_test.columns)

'''
df = pd.DataFrame({'A':[1,2,3,4], 'B':[4,3,2,1], 'C':[1,1,1,1]})
print(df)
s=df.mean()
print(s)
print(s[2])

df = pd.DataFrame({'A':[2,np.nan,3,4], 'B':[4,3,np.nan,1], 'C':[np.nan,1,1,1]})
print(df)
s=df.mean()
print(s)
print(s[2])
s = df.fillna(value=s)
print(s)
'''


