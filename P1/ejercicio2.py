# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################

import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools

#-------------------------------------------------------------------------------#
#---------------------- Ejercicio sobre regresión lineal -----------------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#


# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []
	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(1)
			else:
				y.append(-1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y


	
# Funcion para calcular el error
def Err(x,y,w):
        error = 0
        for i in range(0, len(x)):
                #Calculo el error total como la media de los errores al cuadrado
                #Suma del producto escalar de w por x (función que calculamos para hallar las etiquetas) menos la etiqueta correcta.
                #error = error + (w[0]*x[i][0] + w[1]*x[i][1] + w[2]*x[i][2]-y[i])**2
                error = error + (np.dot(w, x[i])-y[i])**2

        return error/len(x)


	
# Gradiente Descendente Estocastico
def sgd(x, y, lr, max_iters, tam_minibatch):

        #Inicializa el punto inicial a (0,0)
        w = np.array([])
        for i in range(0, len(x[0])):
                w = np.append(w, 0.0)

        w = np.array(w, np.float64)

        for l in range(0, max_iters): #10 secuencias distintas de minibatch
                #Inicializo los minibatch
                minibatch = []

                #Barajo los datos del vector x y las etiquetas en el mismo orden
                c = list(zip(x, y))
                random.shuffle(c)
                x, y = zip(*c)

                #Creo el minibatch (cojo los primeros datos del vector x)
                i = 0
                for i in range(0, tam_minibatch):
                        minibatch.append(x[i])

                #Algoritmo de Gradiente descendente
                suma = np.array([])
                for i in range(0, len(x[0])):
                        suma = np.append(w, 0.0)

                suma = np.array(w, np.float64)

                #Calculo la derivada de Ein, que es la sumatoria de x_i*(<x_i, w> - y_i), donde <,> denota producto escalar
                for k in range(0,tam_minibatch):
                        #print(k)
                        #print("x: " + str(x[k]))
                        suma = suma + minibatch[k]*(np.dot(w, minibatch[k])-y[k])

                #Actualiazción de w según el algoritmo
                w = w - lr*suma
                #print(suma)
                #print(w)
                        
        return w


	
# Algoritmo pseudoinversa	
def pseudoinverse(x, y):
        #Devuelve el producto de la matriz pseudoinversa por y
	return np.dot(np.linalg.pinv(x), y)


	
# Lectura de los datos de entrenamiento
x, y = readData("datos/X_train.npy", "datos/y_train.npy")
# Lectura de los datos para el test
x_test, y_test = readData("datos/X_test.npy", "datos/y_test.npy")


print ('EJERCICIO SOBRE REGRESION LINEAL\n')
print ('Ejercicio 1\n')
# Gradiente descendente estocastico

w = sgd(x, y, 0.001, 100, 64)
#print("w= " + str(w))

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test,y_test,w))

input("\n--- Pulsar tecla para continuar ---\n")

# Algoritmo Pseudoinversa

w = pseudoinverse(x, y)
#print("w= " + str(w))

print ('\nBondad del resultado para el algoritmo de la pseudoinversa:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test,y_test,w))




#Pintamos la gráfica
def grafica(x, y, w):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Ponemos nombre a los ejes
        ax.set_xlabel('Intensidad promedio', fontsize = 10)
        ax.set_ylabel('Simetría', fontsize = 10)
        ax.set_zlabel('Etiqueta: \n 1 si es un 5 \n -1 si es un 1', fontsize = 10)

        #Usamos esto para poner una leyenda, ya que por lo visto
        #la función leyenda no admite el tipo 3D devuelto por scatter
        #y se debe crear una representación ficticia con las mismas
        #características y ponerla en la leyenda
        scatter1_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
        scatter2_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='m', marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['Datos usados para el ajuste (Train set)', 'Hiperplano resultante'], numpoints = 1)

        coor_x = [] #Coordenada de intensidad
        coor_y = [] #Coordenada de simetría
        y_dato = [] #Etiqueta del dato (su imagen por la función f desconocida)

        #Guardamos en vectores los distintos valores
        for i in range(0, len(x)):
                coor_x.append(x[i][1])
                coor_y.append(x[i][2])
                y_dato.append(y[i])

        #Pintamos los datos de entrenamiento
        ax.scatter(coor_x, coor_y, y_dato)
        
        #Creamos una cuadrícula sobre la que se dibuja el hiperplano
        
        #Límites de la cuadrícula
        min_x = np.min(coor_x)
        min_y = np.min(coor_y)
        max_x = np.max(coor_x)
        max_y = np.max(coor_y)

        #Creamos la cuadricula
        cx = np.array([[min_x, max_x], [min_x, max_x]])
        cy = np.array([[min_y, min_y], [max_y, max_y]])

        #Imagen por la función h que hemos calculado de los datos de la cuadrícula
        im_dato = np.array(w[0] + w[1]*cx + w[2]*cy)
        
        #Pintamos el hiperplano w
        ax.plot_surface(cx, cy, im_dato, color="m")
        
        plt.show()

       

input("\n--- Pulsar tecla para continuar ---\n")
print("Gráfica con los resultados obtenidos")
grafica(x, y, w)


#------------------------------Ejercicio 2 -------------------------------------#

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):       
        return np.random.uniform(-size, size, size=(N, 1, d));
	
# EXPERIMENTO	
# a) Muestra de entrenamiento N = 1000, cuadrado [-1,1]x[-1,1]	

print ('\nEjercicio 2\n')
print ('Muestra N = 1000, cuadrado [-1,1]x[-1,1]')

#Muestra:
datos = simula_unif(1000, 2, 1)


#Dibuja los datos de la muestra
def muestra_datos(datos):
        coor_x = [] #Coordenada X de los datos
        coor_y = [] #Coordenada Y de los datos

        #Relleno los vectores de coordenadas
        for i in range(len(datos)):
                coor_x.append(datos[i][0][0])
                coor_y.append(datos[i][0][1])
                
        #Representamos las dos últimas características de x
        plt.scatter(coor_x, coor_y, c='red', label = 'Muestra de entrenamiento')

        #Añado el título 
        plt.title('Muestra uniforme')
        #Añado la leyenda
        plt.legend(loc=2)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()

  
#Muestra los datos
muestra_datos(datos)

#--------------------------------------------------------------------
#b) Asignamos etiquetas a la muestra y pintamos mapa de etiquetas con ruido

#Función que asigna etiquetas
def f(x1, x2):
        return np.sign((x1-0.2)**2 + x2**2 - 0.6)


#Asigna etiquetas a los datos con un 10% de ruido
def asigno_etiquetas(datos):

        #Vector con etiquetas de los datos
        im_datos = []
        
        #Mezclo los datos
        random.shuffle(datos)

        #Numero de datos con ruido: 10% 
        ruido = int(0.1*len(datos))

        #Meto las etiquetas en el vector im_datos
        for i in range(0, len(datos)):
                #Datos con ruido
                if i<ruido:
                        im_datos.append(-f(datos[i][0][0],datos[i][0][1]))
                #Datos sin ruido
                else:
                        im_datos.append(f(datos[i][0][0], datos[i][0][1]))

        return datos, im_datos


#Dibuja la gráfica de los datos "datos" con las etiquetas "im_datos"
def dibuja_grafica(datos, im_datos):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Ponemos nombre a los ejes
        ax.set_xlabel('Coordenada X', fontsize = 10)
        ax.set_ylabel('Coordenada Y', fontsize = 10)
        ax.set_zlabel('Etiqueta: \n Signo de la función f', fontsize = 10)

        #Leyenda
        scatter1_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='c', marker = 'o')
        scatter2_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='m', marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['Etiqueta 1', 'Etiqueta -1'], numpoints = 1)

        #Datos con etiqueta 1
        azul_x = [] #Coordenada X
        azul_y = [] #Coordenada Y
        im_azul = [] #Imagen de los azules (1)
        
        #Datos con etiqueta -1
        rosa_x = [] #Coordenada X
        rosa_y = [] #Coordenada Y
        im_rosa = [] #Imagen de los rosas (-1)

        #Guardamos en vectores los distintos valores
        for i in range(0, len(datos)):
                if im_datos[i] == 1:
                        azul_x.append(datos[i][0][0])
                        azul_y.append(datos[i][0][1])
                        im_azul.append(im_datos[i])
                else:
                        rosa_x.append(datos[i][0][0])
                        rosa_y.append(datos[i][0][1])
                        im_rosa.append(im_datos[i])

        #Pintamos los datos de entrenamiento
        ax.scatter(azul_x, azul_y, im_azul, c='c')
        ax.scatter(rosa_x, rosa_y, im_rosa, c='m')
        
        plt.show() 

#Obtengo los datos barajados con sus etiquetas y dibujo la gráfica
datos, im_datos = asigno_etiquetas(datos)
print ('\nGráfica de los datos con sus etiquetas\n')
dibuja_grafica(datos, im_datos)

# -------------------------------------------------------------------

# c) Ajustar el modelo de regresión lineal

#Crea un vector con el formato que acepta fa función de Gradiente descendente
def formato_datos(datos):
        x=[]# Vector con datos en el formato adecuado
        
        #Añadimos los datos en el formato adecuado al vector x
        for i in range(0, len(datos)):
                x.append(np.array([1, datos[i][0][0], datos[i][0][1]]))
        x = np.array(x, np.float64)
        return x


#Ponemos los datos en el formato adecuado
x = formato_datos(datos)
#Aplicamos el algoritmo de gradiente descendente
print ('\nAplicamos modelo de regresión lineal:\n')
w = sgd(x, im_datos, 0.01, 100, 64)
print ('\nLos pesos w son: ' + str(w) + "\n")

print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,im_datos,w))

#Pintamos la gráfica
def grafica_plano(x, y, w):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Ponemos nombre a los ejes
        ax.set_xlabel('Coordenada X', fontsize = 10)
        ax.set_ylabel('Coordenada Y', fontsize = 10)
        ax.set_zlabel('Signo de la función f', fontsize = 10)

        #Leyenda
        scatter1_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
        scatter2_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='m', marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['Datos usados para el ajuste (Train set)', 'Hiperplano resultante'], numpoints = 1)

        coor_x = [] #Coordenada de intensidad
        coor_y = [] #Coordenada de simetría
        y_dato = [] #Etiqueta del dato (su imagen por la función f desconocida)

        #Guardamos en vectores los distintos valores
        for i in range(0, len(x)):
                coor_x.append(x[i][1])
                coor_y.append(x[i][2])
                y_dato.append(y[i])

        #Pintamos los datos de entrenamiento
        ax.scatter(coor_x, coor_y, y_dato)
        
        #Creamos una cuadrícula sobre la que se dibuja el hiperplano
        
        #Límites de la cuadrícula
        min_x = np.min(coor_x)
        min_y = np.min(coor_y)
        max_x = np.max(coor_x)
        max_y = np.max(coor_y)

        #Creamos la cuadricula
        cx = np.array([[min_x, max_x], [min_x, max_x]])
        cy = np.array([[min_y, min_y], [max_y, max_y]])

        #Imagen por la función h que hemos calculado de los datos de la cuadrícula
        im_dato = np.array(w[0] + w[1]*cx + w[2]*cy)
        
        #Pintamos el hiperplano w
        ax.plot_surface(cx, cy, im_dato, color="m")
        
        plt.show()


input("\n--- Pulsar tecla para continuar ---\n")
print("Gráfica con los resultados de la regresión con ajuste lineal:")
grafica_plano(x, im_datos, w)
# -------------------------------------------------------------------

# d) Ejecutar el experimento 1000 veces

def experimento(max_iters):
        Ein = 0 #Suma de errores de todas las iteraciones
        Eout = 0 #Suma de errores de todas las iteraciones
        
        for i in range(0, max_iters):
                #Genero la muestra y le asigno sus etiquetas
                datos = simula_unif(1000, 2, 1)
                datos, im_datos = asigno_etiquetas(datos)

                #Genero datos para testear el resultado y le asigno sus etiquetas
                test = simula_unif(1000, 2, 1)
                test, im_test = asigno_etiquetas(test)
                #Pongo los datos test en el formato adecuado
                test = formato_datos(test)

                #Aplicamos el algoritmo de gradiente descendente con los datos en el formato adecuado
                x = formato_datos(datos)
                w = sgd(x, im_datos, 0.01, 100, 64)

                #Calculamos el error acumulado
                Ein = Ein + Err(x,im_datos,w)
                Eout = Eout + Err(test, im_test, w)

        return Ein, Eout, x, im_datos, w


n=1000 #Número de iteraciones del experimento
Ein, Eout, x, y, w = experimento(n)
Ein_media = Ein/n      #Error medio en los datos usados para el ajuste
Eout_media = Eout/n    #Error medio en los datos de prueba

print ('\nErrores Ein y Eout medios tras 1000reps del experimento con un ajuste lineal:\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)

input("\n--- Pulsar tecla para continuar ---\n")
print("Gráfica con los resultados de la regresión con ajuste lineal:")
grafica_plano(x, y, w)

# -------------------------------------------------------------------

# Repetición del experimento con ajuste cuadrático en x

#Ajustar los datos al vector de características (1, x1, x2, x1, x2, x1^2, x2^2)
def ajuste_datos(datos):
        x=[]# Vector con datos en el formato adecuado
        
        #Añadimos los datos en el formato adecuado al vector x
        for i in range(0, len(datos)):
                x.append(np.array([1, datos[i][0][0], datos[i][0][1], datos[i][0][0]*datos[i][0][1], datos[i][0][0]**2, datos[i][0][1]**2]))
        x = np.array(x, np.float64)
        return x
 
#print(ajuste_datos(datos)[0])

#Experimento con el vector de características (1, x1, x2, x1, x2, x1^2, x2^2)
def experimento_2(max_iters):
        Ein = 0 #Suma de errores de todas las iteraciones
        Eout = 0 #Suma de errores de todas las iteraciones
        
        for i in range(0, max_iters):
                #Genero la muestra y le asigno sus etiquetas
                datos = simula_unif(1000, 2, 1)
                datos, im_datos = asigno_etiquetas(datos)

                #Genero datos para testear el resultado y le asigno sus etiquetas
                test = simula_unif(1000, 2, 1)
                test, im_test = asigno_etiquetas(test)
                #Pongo los datos test en el formato adecuado
                test = ajuste_datos(test)

                #Aplicamos el algoritmo de gradiente descendente con los datos en el formato adecuado
                x = ajuste_datos(datos)
                w = sgd(x, im_datos, 0.01, 100, 64)

                #Calculamos el error acumulado
                Ein = Ein + Err(x,im_datos,w)
                Eout = Eout + Err(test, im_test, w)

        return Ein, Eout, x, im_datos, w


n=1000 #Número de iteraciones del experimento
Ein, Eout, x, y, w = experimento_2(n)
Ein_media = Ein/n      #Error medio en los datos usados para el ajuste
Eout_media = Eout/n    #Error medio en los datos de prueba
        
print ('\nErrores Ein y Eout medios tras 1000reps del experimento con un ajuste no lineal :\n')
print ("Ein media: ", Ein_media)
print ("Eout media: ", Eout_media)
print ('\nLos pesos w son: ' + str(w) + "\n")


#Pintamos la gráfica
def grafica_parabola(x, y, w):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #Ponemos nombre a los ejes
        ax.set_xlabel('Coordenada X', fontsize = 10)
        ax.set_ylabel('Coordenada Y', fontsize = 10)
        ax.set_zlabel('Signo de la función f', fontsize = 10)

        #Leyenda
        scatter1_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='b', marker = 'o')
        scatter2_proxy = mpl.lines.Line2D([0],[0], linestyle="none", c='m', marker = 'v')
        ax.legend([scatter1_proxy, scatter2_proxy], ['Datos usados para el ajuste (Train set)', 'Hiperplano resultante'], numpoints = 1)

        coor_x = [] #Coordenada de intensidad
        coor_y = [] #Coordenada de simetría
        y_dato = [] #Etiqueta del dato (su imagen por la función f desconocida)

        #Guardamos en vectores los distintos valores
        for i in range(0, len(x)):
                coor_x.append(x[i][1])
                coor_y.append(x[i][2])
                y_dato.append(y[i])

        #Pintamos los datos de entrenamiento
        ax.scatter(coor_x, coor_y, y_dato)
        
        #Creamos una cuadrícula sobre la que se dibuja el hiperplano
        
        #Límites de la cuadrícula
        min_x = np.min(coor_x)
        min_y = np.min(coor_y)
        max_x = np.max(coor_x)
        max_y = np.max(coor_y)

        #Creamos la cuadricula
        '''
        cx_v = np.linspace(min_x, max_x, 100)
        cy_v = np.linspace(min_y, max_y, 100)

        cx=np.array([])
        for i in range(0, 100):
                np.append(cx, cx_v)

        cy=np.array([])
        for i in range(0, 100):
                aux = [cy_v[i]]*100
                np.append(cy, np.array(aux))
        '''
        
        cx, cy = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100))
         
        #Imagen por la función h que hemos calculado de los datos de la cuadrícula
        im_dato = np.array(w[0] + w[1]*cx + w[2]*cy + w[3]*cx*cx + w[4]*cx*cx + w[5]*cy*cy)
        
        #Pintamos el hiperplano w
        ax.plot_surface(cx, cy, im_dato, color="m")
        
        plt.show()

input("\n--- Pulsar tecla para continuar ---\n")
print("Gráfica con los resultados de la regresión con ajuste no lineal:")
grafica_parabola(x,y,w)
