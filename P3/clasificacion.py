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

# Fijamos la semilla
np.random.seed(1)

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

def formato_im(y):
        #Cambio cada etiqueta i por el vector e_i
        im_datos = []

        for i in range(0, len(y)):
                aux = [0 for i in range(0, 9)] #Vector auxiliar con 9 ceros
                aux = np.array(aux)
                aux = np.insert(aux, int(y[i]), 1) #Inserto un 1 en la posición correspondiente
                #print(aux)
                #print(y_train[i])
                        
                im_datos.append(aux) #Añado el vector a la lista de etiquetas
        y = np.array(im_datos)
        #print(y)
        return y

#------------------AJUSTE DEL MODELO--------------------------
#Función sigmoide
def sigma(x):
    return 1/(np.exp(-x)+1)

#Error clasificacion multietiqueta
def Err(x,y,w):
        error = 0
        
        for i in range(0, len(x)):
                #Calculo el error total como el numero de elementos mal clasificados
                for j in range(0, len(w)):
                        error = error - y[i][j]*m.log(sigma(np.dot(w[j],x[i])))
        return error/len(x)

# Regresion Logistica usando Gradiente Descendente Estocastico
def rl_sgd(x, y, lr, max_iters, tam_minibatch):

        w_list = [] #lista con todos los w de la RL
        #Hay que hacer RL binaria 10 veces.
        for k in range(0,10):
                                                   
                #Inicializa el punto inicial a (0,0)
                w = [0 for i in range(0, len(x[0]))]
                w = np.array(w, np.float64)

                iters = 0
                #Paro cuando ||w^t - w^(t-1)||<0.01
                while iters < max_iters:
                
                        #Elijo índices para formar el minibatch
                        m = np.random.randint(0, len(x), size=tam_minibatch)
                        
                        #Creo el minibatch
                        minibatch_x = []
                        minibatch_y = []
                        for i in range(0, tam_minibatch):
                                minibatch_x.append(x[m[i]])
                                minibatch_y.append(y[m[i]])

                        #Algoritmo de Gradiente descendente
                        for i in range(0, tam_minibatch):

                                #Actualiazción de w según el algoritmo
                                suma = np.array([0 for i in range(0, len(x[0]))])
                                for n in range(0, tam_minibatch):
                                        #print((sigma(np.dot(w,x[n])) - y[n][k])*x[n])
                                        aux = np.array((sigma(np.dot(w,minibatch_x[n])) - minibatch_y[n][k])*minibatch_x[n])
                                        suma = suma + aux

                                w = w - lr*suma
                        
                        iters += 1

                w_list.append(w)
                print(iters)
                        
        return w_list

def softmax(x,y,w):

        i = np.random.randint(0, len(x))
        elem = x[i]

        print("La etiqueta es: " + str(y[i]))

        suma = 0
        for i in range(0, len(w)):
                suma += m.exp(np.dot(w[i], elem))

        p = []

        for i in range(0, len(w)):
                p.append(m.exp(np.dot(w[i], elem))/suma)

        p = np.array(p)

        et = np.argmax(p)
        prob = np.amax(p)

        print("La etiqueta que predice es: " + str(et))
        print("Con probabilidad: " + str(prob))

#------------------MÉTRICA DE ERROR--------------------------

def accuracy(x,y,w):

        accuracy = 0
        for i in range(0, len(x)):

                #print("La etiqueta es: " + str(y[i]))

                #Calculo el denominador de P(Cj|x) de SOFTMAX
                suma = 0
                for k in range(0, len(w)):
                        suma += m.exp(np.dot(w[k], x[i]))

                #Añado en una lista las probabilidades de que el
                #elemento de x pertenezca a cada clase
                p = []
                for k in range(0, len(w)):
                        p.append(m.exp(np.dot(w[k], x[i]))/suma)

                p = np.array(p)

                #Su etiqueta es la clase con mayor probabilidad de pertenencia
                et = np.argmax(p)

                #Calculo el total de elementos acertados
                if int(et) == int(y[i]):
                        accuracy += 1
                        
                #prob = np.amax(p)
                #print("La etiqueta que predice es: " + str(et))
                #print("Con probabilidad: " + str(prob))

        return accuracy/len(x)

#------------------REGULARIZACIÓN--------------------------

#Error regularización multietiqueta
def Err_reg(x,y,w):
        error = 0
        
        for i in range(0, len(x)):
                #Calculo el error total como el numero de elementos mal clasificados
                for j in range(0, len(w)):
                        error = error - y[i][j]*m.log(sigma(np.dot(w[j],x[i])))
        return error/len(x) + np.linalg.norm(w)**2


# Regresion Logistica con regularización
def rl_reg(x, y, lr, max_iters, tam_minibatch, landa):

        w_list = [] #lista con todos los w de la RL
        #Hay que hacer RL binaria 10 veces.
        for k in range(0,10):
                                                   
                #Inicializa el punto inicial a (0,0)
                w = [0 for i in range(0, len(x[0]))]
                w = np.array(w, np.float64)

                iters = 0
                #Paro cuando ||w^t - w^(t-1)||<0.01
                while iters < max_iters:
                
                        #Elijo índices para formar el minibatch
                        m = np.random.randint(0, len(x), size=tam_minibatch)
                        
                        #Creo el minibatch
                        minibatch_x = []
                        minibatch_y = []
                        for i in range(0, tam_minibatch):
                                minibatch_x.append(x[m[i]])
                                minibatch_y.append(y[m[i]])

                        #Algoritmo de Gradiente descendente
                        for i in range(0, tam_minibatch):

                                #Actualiazción de w según el algoritmo
                                suma = np.array([0 for i in range(0, len(x[0]))])
                                for n in range(0, tam_minibatch):
                                        #print((sigma(np.dot(w,x[n])) - y[n][k])*x[n])
                                        aux = np.array((sigma(np.dot(w,minibatch_x[n])) - minibatch_y[n][k])*minibatch_x[n])
                                        suma = suma + aux

                                w = w - lr*(suma + landa*w)
                        
                        iters += 1

                w_list.append(w)
                print(iters)
                        
        return w_list


#------------------VALIDACIÓN--------------------------

def validacion_cruzada(x, y, n, lr, max_iters, tam_minibatch):
        e_cv = 0

        #Hago validación cruzada con n conjuntos diferentes 
        l = len(x)
        l = int(l/n)

        datos = []
        im_datos = []
        datos_val = []
        im_datos_val = []
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
        
        #print("Shape datos: " + str(datos.shape))
        #print("Shape im_datos: " + str(im_datos.shape))
        #print("Shape datos val: " + str(datos_val.shape))
        #print("Shape im_datos val: " + str(im_datos_val.shape))

        #print("Longitud de x:" + str(len(x)))
        #print("Número de conjuntos:" + str(n))
        #print("Tamaño de los conjuntos:" + str(l))

        #Ajusto el modelo
        for i in range(0,n):
                #print("Datos: " + str(len(datos[i])))
                #print(len(im_datos[i]))
                w = rl_sgd(datos[i], im_datos[i], lr, max_iters, tam_minibatch)
                e_val = Err(datos_val[i],im_datos_val[i],w) #Calculo el error con los datos de validación
                e_cv += e_val #Error acumulado

        return e_cv/n

#------------------GRAFICAS--------------------------
def grafica(datos, im_datos, titulo):

        #Aplico PCA para quedarme solo con dos componentes
        pca = PCA(n_components = 2, random_state=1)
        pca.fit(datos)
        datos = pca.transform(datos)
        
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
                        pos0_x.append(datos[i][0])
                        pos0_y.append(datos[i][1])
                elif im_datos[i] == 1:
                        #Vector de datos con etiquetas 1
                        pos1_x.append(datos[i][0])
                        pos1_y.append(datos[i][1])
                elif im_datos[i] == 2:
                        #Vector de datos con etiquetas 2
                        pos2_x.append(datos[i][0])
                        pos2_y.append(datos[i][1])
                elif im_datos[i] == 3:
                        #Vector de datos con etiquetas 3
                        pos3_x.append(datos[i][0])
                        pos3_y.append(datos[i][1])
                elif im_datos[i] == 4:
                        #Vector de datos con etiquetas 4
                        pos4_x.append(datos[i][0])
                        pos4_y.append(datos[i][1])
                elif im_datos[i] == 5:
                        #Vector de datos con etiquetas 5
                        pos5_x.append(datos[i][0])
                        pos5_y.append(datos[i][1])
                elif im_datos[i] == 6:
                        #Vector de datos con etiquetas 6
                        pos6_x.append(datos[i][0])
                        pos6_y.append(datos[i][1])
                elif im_datos[i] == 7:
                        #Vector de datos con etiquetas 7
                        pos7_x.append(datos[i][0])
                        pos7_y.append(datos[i][1])
                elif im_datos[i] == 8:
                        #Vector de datos con etiquetas 8
                        pos8_x.append(datos[i][0])
                        pos8_y.append(datos[i][1])
                else:
                        #Vector de datos con etiquetas 9
                        pos9_x.append(datos[i][0])
                        pos9_y.append(datos[i][1])
        
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
x, y = readData("datos/optdigits.tra")
x_test, y_test = readData("datos/optdigits.tes")

#Establezco la longitud del conjunto de validación 
l = len(x)
l = l/4
l = int(l)

#Actualizo conjunto train y conjunto de validación
x_train = x[:-l]
y_train = y[:-l]
x_validation = x[-l:]
y_validation = y[-l:]

#CARACTERÍSTICAS LINEALES

#Preprocesado de datos con PCA
#Conjunto training
#Se queda con las características que explican el 99% de la distribución.
pca = PCA(n_components = 0.99, random_state=1) 
pca.fit(x_train)
x_train = pca.transform(x_train)
#Escala los datos
scaler = proc.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#Le aplico la misma transformación al conjunto de validación
x_validation = pca.transform(x_validation)
x_validation = scaler.transform(x_validation)

#Le aplico la misma transformación al conjunto test
x_test = pca.transform(x_test)
x_test = scaler.transform(x_test)

#Le aplico la misma transformación al archivo de train (para validar despés)
x = pca.transform(x)
x = scaler.transform(x)

#Graficas
#grafica(x_train, y_train, "Dígitos")
#grafica(x_validation, y_validation, "Dígitos")

#Ajusto el formato de las etiquetas a vectores [0..0 1 0..0]
y_train_v = formato_im(y_train)
y_val_v = formato_im(y_validation)
y_test_v = formato_im(y_test)
y_v = formato_im(y)

'''
#Calculo w por medio de la regresión logística
w = rl_sgd(x_train, y_train_v, 0.005, 100, 32)

#Error de los 3 conjuntos diferentes
e_tra = Err(x_train,y_train_v, w)
e_val = Err(x_validation, y_val_v, w)
e_test = Err(x_test, y_test_v, w)

print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))


print("La precisión es de: " + str(accuracy(x_train,y_train,w)))
print("La precisión es de: " + str(accuracy(x_validation,y_validation,w)))
'''

#CARACTERÍSTICAS CUADRÁTICAS:

#Añado características cuadráticas a los datos
poly = proc.PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_validation = poly.transform(x_validation)
x_test = poly.transform(x_test)
x = poly.transform(x)

#Preprocesado de datos con PCA
#Conjunto training
#Se queda con las características que explican el 99% de la distribución.
pca = PCA(n_components = 0.99, random_state=1) 
pca.fit(x_train)
x_train = pca.transform(x_train)
#Escala los datos
scaler = proc.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

#Le aplico la misma transformación al conjunto de validación
x_validation = pca.transform(x_validation)
x_validation = scaler.transform(x_validation)

#Le aplico la misma transformación al conjunto test
x_test = pca.transform(x_test)
x_test = scaler.transform(x_test)

#Le aplico la misma transformación al archivo de train (para validar despés)
x = pca.transform(x)
x = scaler.transform(x)

'''
#Calculo w por medio de la regresión logística
w = rl_sgd(x_train, y_train_v, 0.005, 100, 32)

#Error de los 3 conjuntos diferentes
e_tra = Err(x_train,y_train_v, w)
e_val = Err(x_validation, y_val_v, w)
e_test = Err(x_test, y_test_v, w)

print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))

print("La precisión es de: " + str(accuracy(x_train,y_train,w)))
print("La precisión es de: " + str(accuracy(x_validation,y_validation,w)))
'''

#MODELO CUADRÁTICO REGULARIZADO 
'''
#Calculo w por medio de la regresión logística
w = rl_reg(x_train, y_train_v, 0.005, 100, 32, 0.0001)

#Error de los 3 conjuntos diferentes
e_tra = Err_reg(x_train,y_train_v, w)
e_val = Err_reg(x_validation, y_val_v, w)
e_test = Err_reg(x_test, y_test_v, w)

print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))

print("La precisión es de: " + str(accuracy(x_train,y_train,w)))
print("La precisión es de: " + str(accuracy(x_validation,y_validation,w)))
'''

#CALCULO AJUSTE CON MODELO ELEGIDO
w = rl_sgd(x, y_v, 0.005, 100, 32)
e = Err(x_test, y_test_v, w)
print("El error del conjunto de test es: " + str(e))
print("La precisión es de: " + str(accuracy(x_test,y_test,w)))

e_cv = validacion_cruzada(x, y_v, 5, 0.005, 100, 32)
print("El error de validación cruzada es: " + str(e_cv))

