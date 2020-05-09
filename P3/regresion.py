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

#Ajusta el formato del vector de datos para usar modelo lineal
def formato_datos(x):
        datos = []
        for i in range(0,len(x)):
                aux = np.concatenate((np.array([1]), x[i]), axis=0)
                datos.append(aux)
        datos = np.array(datos)
        return datos

#------------------AJUSTE DEL MODELO--------------------------

# Funcion para calcular el error con y un DataFrame
def Err(x,y,w):
        error = 0
        for i in range(0, len(x)):
                #Calculo el error total como la media de los errores al cuadrado
                #print(np.dot(w, x[i]))
                #print(y.iloc[i])
                #print((np.dot(w, x[i])-y[i]))
                error = error + (np.dot(w, x[i])-y.iloc[i])**2

        return error/len(x)

# Funcion para calcular el error con y un vector numpy
def Err_np(x,y,w):
        error = 0
        for i in range(0, len(x)):
                #Calculo el error total como la media de los errores al cuadrado
                #print(np.dot(w, x[i]))
                #print(y.iloc[i])
                #print((np.dot(w, x[i])-y[i]))
                error = error + (np.dot(w, x[i])-y[i])**2

        return error/len(x)
	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
        #Devuelve el producto de la matriz pseudoinversa por y
	return np.dot(np.linalg.pinv(x), y)

#------------------REGULARIZACIÓN--------------------------
	
# Algoritmo pseudoinversa	
def pseudoinverse_reg(x, y, l):
        #Devuelve el producto de la matriz pseudoinversa por y
        #((X^T*X + λI)^{−1}*X^T)y
	return np.dot(np.dot(np.linalg.inv(np.dot(x.T, x) + l*np.identity(len(x[0]))), x.T), y)

#------------------VALIDACIÓN--------------------------

def validacion_cruzada(x, y, n):
        e_cv = 0 #Error validación cruzada acumulado

        #Hago validación cruzada con n conjuntos diferentes 
        l = len(x)
        l = int(l/n)

        datos = []               #Vector de datos para training de cada iteración
        im_datos = []            #Vector de etiquetas para training de cada iteración
        datos_val = []           #Vector de datos para validación de cada iteración
        im_datos_val = []        #Vector de etiquetas para validación de cada iteración
        
        for i in range(0,n):
                #print("Indice inicial:" + str(i*l))
                #print("Indice final:" + str((i+1)*l-1))
                aux_x = np.concatenate((x[:i*l], x[(i+1)*l:]), axis=0)
                aux_y = np.concatenate((y[:i*l], y[(i+1)*l:]), axis=0)
                #print("Aux_x: " + str(len(aux_x)))
                #print("Aux_x: " + str(aux_x.shape))
                datos.append(aux_x) #Valores de entrenamiento
                im_datos.append(aux_y)
                datos_val.append(x[i*l:(i+1)*l]) #Valores de validación
                im_datos_val.append(y[i*l:(i+1)*l])

        datos = np.array(datos)
        im_datos = np.array(im_datos)
        datos_val = np.array(datos_val)
        im_datos_val = np.array(im_datos_val)

        '''
        print("Shape datos: " + str(datos.shape))
        print("Shape im_datos: " + str(im_datos.shape))
        print("Shape datos val: " + str(datos_val.shape))
        print("Shape im_datos val: " + str(im_datos_val.shape))

        print("Longitud de x:" + str(len(x)))
        print("Número de conjuntos:" + str(n))
        print("Tamaño de los conjuntos:" + str(l))
        '''

        #Ajusto el modelo
        for i in range(0,n):
                #print("Datos: " + str(len(datos[i])))
                #print(len(im_datos[i]))
                w = pseudoinverse(datos[i], im_datos[i])
                e_val = Err_np(datos_val[i],im_datos_val[i],w) #Calculo el error con los datos de validación
                e_cv += e_val #Error acumulado

        return e_cv/n

#------------------PRACTICA 3--------------------------

#Leemos los datos del fichero
x, y = readData('datos/communities.data')
#print(x.shape)
#print(y.shape)

#Establezco la longitud del conjunto de los conjuntos de training, validación y test (60%, 20%, 20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train, x_val, y_train, y_val =  train_test_split(x_train, y_train, test_size=0.25)

'''
print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
print(x_test.shape)
print(y_test.shape)
'''

#PREPROCESADO DE DATOS
#Quitamos las 5 caractericticas no predicativas:
x_train = x_train.drop([0,1,2,3,4], axis = 1)
x_val = x_val.drop([0,1,2,3,4], axis = 1)
x_test = x_test.drop([0,1,2,3,4], axis = 1)

#Quitamos características con más del 1% de valores incógnita
l = int(99*len(x_train.index)/100)
x_train = x_train.dropna(thresh = l, axis = 1)
x_val = x_val.reindex(columns = x_train.columns)
x_test = x_test.reindex(columns = x_train.columns)

#Rellenamos los valores incógnita restantes con la media de los valores de la característica en la muestra
x_train = x_train.fillna(value = x_train.mean())
x_val = x_val.fillna(value = x_val.mean())
x_test = x_test.fillna(value = x_test.mean())

#CARACTERÍSTICAS LINEALES

#Aplicamos PCA a los datos(PASA DE DATAFRAME A NUMPY-ARRAY)
pca = PCA(n_components = 0.99, random_state=1) 
pca.fit(x_train)
x_train = pca.transform(x_train)
#Escala los datos
scaler = proc.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#Le aplico la misma transformación al conjunto de validación
x_val = pca.transform(x_val)
x_val = scaler.transform(x_val)

#Le aplico la misma transformación al conjunto test
x_test = pca.transform(x_test)
x_test = scaler.transform(x_test)

#Ajusto el formato de los datos
x_train = formato_datos(x_train)
x_val = formato_datos(x_val)
x_test = formato_datos(x_test)

#Calculo w por medio de la matriz pseudoinversa
w = pseudoinverse(x_train, y_train)

#Error de los 2 conjuntos diferentes
e_tra = Err(x_train, y_train, w)
e_val = Err(x_val, y_val, w)

print("\n CARACTERÍSTICAS LINEALES \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))

#MODELO LINEAL REGULARIZADO

#Calculo w por medio de la matriz pseudoinversa
w = pseudoinverse_reg(x_train, y_train, 0.001)

#Error de los 2 conjuntos diferentes
e_tra = Err(x_train, y_train, w)
e_val = Err(x_val, y_val, w)

print("\n CARACTERÍSTICAS LINEALES CON REGULARIZACIÓN \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))

#Guardamos datos con características lineales en variables para usarlos en el futuro
x_train_l = x_train
x_val_l = x_val
x_test_l = x_test

#CARACTERÍSTICAS CUADRÁTICAS

#Añado características cuadráticas a los datos
poly = proc.PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_val = poly.transform(x_val)
x_test = poly.transform(x_test)

#Aplicamos PCA a los datos(PASA DE DATAFRAME A NUMPY-ARRAY)
pca = PCA(n_components = 0.99, random_state=1) 
pca.fit(x_train)
x_train = pca.transform(x_train)
#print(len(x_train[0]))
#Escala los datos
scaler = proc.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#Le aplico la misma transformación al conjunto de validación
x_val = pca.transform(x_val)
x_val = scaler.transform(x_val)

#Le aplico la misma transformación al conjunto test
x_test = pca.transform(x_test)
x_test = scaler.transform(x_test)

#Ajusto el formato de los datos
x_train = formato_datos(x_train)
x_val = formato_datos(x_val)
x_test = formato_datos(x_test)

#Calculo w por medio de la matriz pseudoinversa
w = pseudoinverse(x_train, y_train)

#Error de los 2 conjuntos diferentes
e_tra = Err(x_train, y_train, w)
e_val = Err(x_val, y_val, w)

print("\n CARACTERÍSTICAS CUADRÁTICAS \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))

#CARACTERÍSTICAS CUADRÁTICAS CON REGULARIZACIÓN

#Calculo w por medio de la matriz pseudoinversa
w = pseudoinverse_reg(x_train, y_train, 172)
#172

#Error de los 2 conjuntos diferentes
e_tra = Err(x_train, y_train, w)
e_val = Err(x_val, y_val, w)

print("\n CARACTERÍSTICAS LINEALES CON REGULARIZACIÓN \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))

#MODELO ELEGIDO
print("\n MODELO ELEGIDO \n")

'''
print(len(x_train_l))
print(len(x_val_l))
print(len(x_train_l[0]))
'''

#Ajustamos el modelo con todos los datos de entrenamiento
x = np.concatenate((x_train_l, x_val_l), axis = 0)
y = np.concatenate((y_train, y_val), axis = 0)

'''
print(len(x))
print(len(x[0]))
'''

w = pseudoinverse(x_train_l, y_train)
e = Err_np(x,y,w)
print("El error del conjunto test es: " + str(e))

#Calculamos la cote del error por validación cruzada
e_cv = validacion_cruzada(x, y, 5)
print("El error de validación cruzada es: " + str(e_cv))


