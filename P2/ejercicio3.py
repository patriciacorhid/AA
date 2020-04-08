# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

# Fijamos la semilla
np.random.seed(1)

#------------------------------Ejercicio 2 -------------------------------------#

print ('\n BONUS \n')
print ('\nEjercicio 1\n')

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 4 o la 8
	for i in range(0,datay.size):
		if datay[i] == 4 or datay[i] == 8:
			if datay[i] == 8:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


# Algoritmo pseudoinversa (REGRESION LINEAL)	
def pseudoinverse(x, y):
        #Devuelve el producto de la matriz pseudoinversa por y
	return np.dot(np.linalg.pinv(x), y)

#Dibuja los datos de la nube de puntos
def grafica(datos, im_datos, w, titulo):
        pos_x = [] #Coordenada X de los datos con etiqueta 1
        pos_y = [] #Coordenada Y de los datos con etiqueta 1
        neg_x = [] #Coordenada X de los datos con etiqueta -1
        neg_y = [] #Coordenada Y de los datos con etiqueta -1

        #Relleno los vectores de coordenadas
        for i in range(len(datos)):
                if im_datos[i] == 1:
                        #Vector de datos con etiquetas positivas
                        pos_x.append(datos[i][1])
                        pos_y.append(datos[i][2])
                else:
                        #Vector de datos con etiquetas positivas
                        neg_x.append(datos[i][1])
                        neg_y.append(datos[i][2])

        #Limites
        min_posx = np.min(pos_x)
        max_posx = np.max(pos_x)

        min_negx = np.min(neg_x)
        max_negx = np.max(neg_x)

        rango = [np.min([min_posx, min_negx]), np.max([max_posx, max_negx])]
        
        #Representamos los datos
        plt.scatter(pos_x, pos_y, c='red', label = 'Puntos con etiqueta positiva')
        plt.scatter(neg_x, neg_y, c='b', label = 'Puntos con etiqueta negativa')

        X = np.linspace(rango[0], rango[1], 100, endpoint=True)
        Y = -(w[0] + w[1]*X)/w[2]
        plt.plot(X, Y, color="black")

        #Añado el título 
        plt.title(titulo)
        #Añado la leyenda
        plt.legend(loc=4)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()


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


# Lectura de los datos de entrenamiento
x, y = readData("datos/X_train.npy", "datos/y_train.npy")
# Lectura de los datos para el test
x_test, y_test = readData("datos/X_test.npy", "datos/y_test.npy")

#Ejecutamos la regresion lineal
w = pseudoinverse(x, y)
print('Los pesos obtenidos en la regresión lineal son: ' + str(w))
print('Grafica que muestra el resultado de la regresión lineal')
grafica(x, y, w, 'Regresión lineal')
print('El error Ein de la recta conseguida en la regresión es: ' + str(Err(x,y,w)) + '\n')

#a)--------------------------------------------------------

w_train = PLA_pocket(x, y, 50, w)
print('Los pesos obtenidos con PLA-pocket con los datos de entrenamiento son: ' + str(w_train))
print('Grafica que muestra el resultado del PLA-pocket con los datos de entrenamiento\n')
grafica(x, y, w_train, 'PLA-pocket training')

print('Grafica que muestra el resultado del PLA-pocket con los datos de prueba')
grafica(x_test, y_test, w_train, 'PLA-pocket test')

input("\n--- Pulsar tecla para continuar ---\n")

#b)--------------------------------------------------------

print('El error Ein es: ' + str(Err(x,y,w_train)))
print('El error Eout es: ' + str(Err(x_test,y_test,w_train)))

input("\n--- Pulsar tecla para continuar ---\n")
