# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.decomposition import PCA

#------------------PROCESAMIENTO DE DATOS--------------------------

#Leemos los datos del fichero
def readData(file):
	# Leemos los ficheros
        x = []
        y = []
        data = open(file, "r")
        
        for line in data:
                aux = line.split(",")
                x.append(aux[:-1])
                y.append(aux[-1])                

        x = np.array(x, np.float64)
        y = np.array(y, np.float64)
                
        return x, y

#Ajusta el formato del vector de 2 componentes del PCA
def formato_datos(x):
        datos = [] #Datos finales
        for i in range(0, len(x)):
                datos.append([1, x[i][0], x[i][1]])
        return datos

#------------------AJUSTE DEL MODELO--------------------------

#Error clasificacion
def Err(x,y,w):
        error = 0
        for i in range(0, len(x)):
                #Calculo el error total como el numero de elementos mal clasificados
                if np.sign(np.dot(w, x[i])) != y[i]:
                        error = error + 1

        return error/len(x)

#Algoritmo PLA
def ajusta_PLA(datos, label, max_iter, vini):
    w = vini #Vector con pesos del hiperplano
    iters = 0 #Iteraciones necesarias para converger
    sin_cambios = 0 #Veces en las que no se cambio el w

    while sin_cambios < len(datos) and iters < max_iter:
        #Para cada dato de la muestra
        for i in range(0, len(datos)):
            #Si clasifico mal, modifico el w
            if np.sign(np.dot(w,datos[i])) != label[i]:
                w = w + label[i]*datos[i]
                sin_cambios = 0 #He hecho cambios en w
                #Si clasifico bien, guardo que no hago cambios en w
            else:
                sin_cambios = sin_cambios + 1

        #Aumento el numero de iteraciones
        iters = iters +1

    return w

#Algoritmo PLA pocket
def PLA_pocket(datos, label, max_iter, vini):
        w = vini #w optimo
        w_new = vini #w que consigo en las iteraciones
        Ein = Err(datos, label, w) #Error w optimo
        
        for i in range(0, max_iter):
                #Ejecuto PLA para conseguir otro w
                w_new = ajusta_PLA(datos, label, 1, w_new)
                #Calculo el error del nuevo w
                Ein_new = Err(datos, label, w_new)
                #Si su error es menor, actualizo w
                if Ein_new < Ein:
                        w = w_new
                        Ein = Ein_new
        return w

#------------------SELECCION DEL MODELO--------------------------
#Los datos que usaremos serán [1, intensidad promedio, simetría]
def grado1(x):
        return x

def grado2(x):
        datos = [] #Datos finales
        for i in range(0, len(x)):
                a = x[i][1]
                b = x[i][2]
                datos.append([1, a, b, a*b, a**2, b**2])
        return datos

def grado3(x):
        datos = [] #Datos finales
        for i in range(0, len(x)):
                a = x[i][1]
                b = x[i][2]
                datos.append([1, a, b, a*b, a**2, b**2, a**3, b**3, a**2*b, a*b**2])
        return datos

def grado4(x):
        datos = [] #Datos finales
        for i in range(0, len(x)):
                a = x[i][1]
                b = x[i][2]
                datos.append([1, a, b, a*b, a**2, b**2, a**3, b**3, a**2*b, a*b**2, a**4, b**4])
        return datos

def grado5(x):
        datos = [] #Datos finales
        for i in range(0, len(x)):
                a = x[i][1]
                b = x[i][2]
                datos.append([1, a, b, a*b, a**2, b**2, a**3, b**3, a**2*b, a*b**2, a**4, b**4, a**5, b**5])
        return datos

#------------------VALIDACIÓN--------------------------

def validacion_cruzada(x, y):
        e_cv = 0
        for i in range(0, len(x)):
                x = x[i]  #Valores que uso para la validación
                y = y[i]
                x_val = np.delete(x,i) #Valores que uso para entrenar
                y_val = np.delete(y,i)
                vini = np.array([]) #Vector inicial para PLA

                #Inicializo el vector inicial a 0
                for i in range(0, len(x[0])):
                        np.append(vini, 0.0)

                #Ajusto el modelo
                w = PLA_pocket(x_val, y_val, 100, vini)
                e_val = Err(x,y,w) #Calculo el error con el dato de validación
                e_cv += e_val #Error acumulado

        return e_cv/len(x)

#------------------REGULARIZACIÓN--------------------------

#------------------GRAFICAS--------------------------
def grafica(datos, im_datos, titulo):
        pos0_x = [] #Coordenada X de los datos con etiqueta 0
        pos0_y = [] #Coordenada Y de los datos con etiqueta 0
        pos1_x = [] #Coordenada X de los datos con etiqueta 1
        pos1_y = [] #Coordenada Y de los datos con etiqueta 1
        pos2_x = [] #Coordenada X de los datos con etiqueta 2
        pos2_y = [] #Coordenada Y de los datos con etiqueta 2
        pos3_x = [] #Coordenada X de los datos con etiqueta 3
        pos3_y = [] #Coordenada Y de los datos con etiqueta 3
        pos4_x = [] #Coordenada X de los datos con etiqueta 4
        pos4_y = [] #Coordenada Y de los datos con etiqueta 4
        pos5_x = [] #Coordenada X de los datos con etiqueta 5
        pos5_y = [] #Coordenada Y de los datos con etiqueta 5
        pos6_x = [] #Coordenada X de los datos con etiqueta 6
        pos6_y = [] #Coordenada Y de los datos con etiqueta 6
        pos7_x = [] #Coordenada X de los datos con etiqueta 7
        pos7_y = [] #Coordenada Y de los datos con etiqueta 7
        pos8_x = [] #Coordenada X de los datos con etiqueta 8
        pos8_y = [] #Coordenada Y de los datos con etiqueta 8
        pos9_x = [] #Coordenada X de los datos con etiqueta 9
        pos9_y = [] #Coordenada Y de los datos con etiqueta 9

        #Relleno los vectores de coordenadas
        for i in range(len(datos)):
                if im_datos[i] == 0:
                        #Vector de datos con etiquetas 0
                        pos0_x.append(datos[i][1])
                        pos0_y.append(datos[i][2])
                elif im_datos[i] == 1:
                        #Vector de datos con etiquetas 1
                        pos1_x.append(datos[i][1])
                        pos1_y.append(datos[i][2])
                elif im_datos[i] == 2:
                        #Vector de datos con etiquetas 2
                        pos2_x.append(datos[i][1])
                        pos2_y.append(datos[i][2])
                elif im_datos[i] == 3:
                        #Vector de datos con etiquetas 3
                        pos3_x.append(datos[i][1])
                        pos3_y.append(datos[i][2])
                elif im_datos[i] == 4:
                        #Vector de datos con etiquetas 4
                        pos4_x.append(datos[i][1])
                        pos4_y.append(datos[i][2])
                elif im_datos[i] == 5:
                        #Vector de datos con etiquetas 5
                        pos5_x.append(datos[i][1])
                        pos5_y.append(datos[i][2])
                elif im_datos[i] == 6:
                        #Vector de datos con etiquetas 6
                        pos6_x.append(datos[i][1])
                        pos6_y.append(datos[i][2])
                elif im_datos[i] == 7:
                        #Vector de datos con etiquetas 7
                        pos7_x.append(datos[i][1])
                        pos7_y.append(datos[i][2])
                elif im_datos[i] == 8:
                        #Vector de datos con etiquetas 8
                        pos8_x.append(datos[i][1])
                        pos8_y.append(datos[i][2])
                else:
                        #Vector de datos con etiquetas 9
                        pos9_x.append(datos[i][1])
                        pos9_y.append(datos[i][2])
        
        #Representamos los datos
        plt.scatter(pos0_x, pos0_y, c='r', label = '0')
        plt.scatter(pos1_x, pos1_y, c='darkgreen', label = '1')
        plt.scatter(pos2_x, pos2_y, c='navy', label = '2')
        plt.scatter(pos3_x, pos3_y, c='c', label = '3')
        plt.scatter(pos4_x, pos4_y, c='deeppink', label = '4')
        plt.scatter(pos5_x, pos5_y, c='gold', label = '5')
        plt.scatter(pos6_x, pos6_y, c='teal', label = '6')
        plt.scatter(pos7_x, pos7_y, c='lime', label = '7')
        plt.scatter(pos8_x, pos8_y, c='purple', label = '8')
        plt.scatter(pos9_x, pos9_y, c='coral', label = '9')
        
        #Añado el título 
        plt.title(titulo)
        #Añado la leyenda
        plt.legend(loc=2)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()

#------------------PRACTICA 3--------------------------

#Leemos los datos del fichero
x_train, y_train = readData("datos/optdigits.tra")
x_test, y_test = readData("datos/optdigits.tes")

#Establezco la longitud del conjunto de validación 
l = len(x_train)
l = l/4
l = int(l)

#Actualizo conjunto test y conjunto de validación
x_train = x_train[:-l]
y_train = y_train[:-l]
x_validation = x_train[-l:]
y_validation = y_train[-l:]

#Preprocesado de datos con PCA

#Conjunto training
pca = PCA(n_components = 2)
pca.fit(x_train)
x_train = pca.transform(x_train)

x_train = formato_datos(x_train)
grafica(x_train, y_train, "Dígitos")

#Conjunto validación
pca.fit(x_validation)
x_validation = pca.transform(x_validation)

x_validation = formato_datos(x_validation)
grafica(x_validation, y_validation, "Dígitos")
