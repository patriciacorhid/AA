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
                #print(iters)
                        
        return w_list

def softmax(x,y,w):
        #np.random.seed(1)
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
        return (error + np.linalg.norm(w)**2)/len(x)


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
                #print(iters)
                        
        return w_list


#------------------VALIDACIÓN--------------------------

def validacion_cruzada(x, y, n, lr, max_iters, tam_minibatch, y_e):
        e_cv = 0
        acc = 0

        #Hago validación cruzada con n conjuntos diferentes 
        l = len(x)
        l = int(l/n)

        datos = []               #Vector de datos para training de cada iteración
        im_datos = []            #Vector de etiquetas para training de cada iteración
        datos_val = []           #Vector de datos para validación de cada iteración
        im_datos_val = []        #Vector de etiquetas para validación de cada iteración
        im_datos_val_e = []      #Vector de etiquetas (formato dígito) para validación de cada iteración
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
                im_datos_val_e.append(y_e[i*l:(i+1)*l])

        datos = np.array(datos)
        im_datos = np.array(im_datos)
        datos_val = np.array(datos_val)
        im_datos_val = np.array(im_datos_val)
        im_datos_val_e = np.array(im_datos_val_e)

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
                w = rl_reg(datos[i], im_datos[i], lr, max_iters, tam_minibatch, 0.0001)
                e_val = Err_reg(datos_val[i],im_datos_val[i],w) #Calculo el error con los datos de validación
                e_cv += e_val #Error acumulado
                acc += accuracy(datos_val[i], im_datos_val_e[i],w) #Accuracy acumulada 

        return e_cv/n, acc/n

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

#Ajusto el formato de las etiquetas a vectores [0..0 1 0..0]
y_train_v = formato_im(y_train)
y_val_v = formato_im(y_validation)
y_test_v = formato_im(y_test)

#Calculo w por medio de la regresión logística
w = rl_sgd(x_train, y_train_v, 0.005, 100, 32)

#Error de los 3 conjuntos diferentes
e_tra = Err(x_train,y_train_v, w)
e_val = Err(x_validation, y_val_v, w)
e_test = Err(x_test, y_test_v, w)

print("\n CARACTERÍSTICAS LINEALES \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))


print("La precisión del conjunto de entrenamiento es: " + str(accuracy(x_train,y_train,w)))
print("La precisión del conjunto de validación es: " + str(accuracy(x_validation,y_validation,w)))

#MODELO LINEAL REGULARIZADO 
#Calculo w por medio de la regresión logística
w = rl_reg(x_train, y_train_v, 0.005, 100, 32, 0.0001)

#Error de los 3 conjuntos diferentes
e_tra = Err_reg(x_train,y_train_v, w)
e_val = Err_reg(x_validation, y_val_v, w)
e_test = Err_reg(x_test, y_test_v, w)

print("\n CARACTERÍSTICAS LINEALES CON REGULARIZACIÓN \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))

print("La precisión del conjunto de entrenamiento es: " + str(accuracy(x_train,y_train,w)))
print("La precisión del conjunto de validación es: " + str(accuracy(x_validation,y_validation,w)))

#CARACTERÍSTICAS CUADRÁTICAS:

#Añado características cuadráticas a los datos
poly = proc.PolynomialFeatures(2)
x_train = poly.fit_transform(x_train)
x_validation = poly.transform(x_validation)
x_test = poly.transform(x_test)

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

#Calculo w por medio de la regresión logística
w = rl_sgd(x_train, y_train_v, 0.005, 100, 32)

#Error de los 3 conjuntos diferentes
e_tra = Err(x_train,y_train_v, w)
e_val = Err(x_validation, y_val_v, w)
e_test = Err(x_test, y_test_v, w)

print("\n CARACTERÍSTICAS CUADRÁTICAS \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))

print("La precisión del conjunto de entrenamiento es: " + str(accuracy(x_train,y_train,w)))
print("La precisión del conjunto de validación es: " + str(accuracy(x_validation,y_validation,w)))

#MODELO CUADRÁTICO REGULARIZADO 
#Calculo w por medio de la regresión logística
w = rl_reg(x_train, y_train_v, 0.005, 100, 32, 0.0001)

#Error de los 3 conjuntos diferentes
e_tra = Err_reg(x_train,y_train_v, w)
e_val = Err_reg(x_validation, y_val_v, w)
e_test = Err_reg(x_test, y_test_v, w)

print("\n CARACTERÍSTICAS CUADRÁTICAS CON REGULARIZACIÓN \n")
print("El error del conjunto de entrenamiento es: " + str(e_tra))
print("El error del conjunto de validación es: " + str(e_val))
print("El error del conjunto test es: " + str(e_test))

print("La precisión del conjunto de entrenamiento es: " + str(accuracy(x_train,y_train,w)))
print("La precisión del conjunto de validación es: " + str(accuracy(x_validation,y_validation,w)))

#CALCULO AJUSTE CON MODELO ELEGIDO

x = np.concatenate((x_train, x_validation), axis=0)
y_v = np.concatenate((y_train_v, y_val_v), axis=0)

print("\n MODELO ELEGIDO \n")
w = rl_reg(x, y_v, 0.005, 100, 32, 0.0001)
e = Err_reg(x,y_v,w)
print("El error del conjunto test es: " + str(e))
print("La precisión del conjunto test es: " + str(accuracy(x_test,y_test,w)))

e_cv, acc = validacion_cruzada(x, y_v, 5, 0.005, 100, 32, y)
print("El error de validación cruzada es: " + str(e_cv))
print("La precisión de validación cruzada es: " + str(acc))

print("\n EJEMPLO: \n")

np.random.seed(1)
for i in range(0, 3):
        softmax(x, y, w)
        print("*******")
