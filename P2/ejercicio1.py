# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

# Fijamos la semilla
np.random.seed(1)

#Devuelve N vectores de dimensión dim dentro del rango
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

#Devuelve N vectores de dimensión dim siguiendo una normal
def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna se usará una N(0,sqrt(5)) y para la segunda N(0,sqrt(7))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
        
    return out

#Devuelve los parametros de una recta y=ax+b que corta al cuadrado intervalo
def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

#------------------------------Ejercicio 1 -------------------------------------#

#Dibuja los datos de la nube de puntos
def muestra_datos(datos, titulo):
        coor_x = [] #Coordenada X de los datos
        coor_y = [] #Coordenada Y de los datos

        #Relleno los vectores de coordenadas
        for i in range(len(datos)):
                coor_x.append(datos[i][0])
                coor_y.append(datos[i][1])
                
        #Representamos los datos
        plt.scatter(coor_x, coor_y, c='red', label = 'Nube de puntos')

        #Añado el título 
        plt.title(titulo)
        #Añado la leyenda
        plt.legend(loc=2)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()

print ('EJERCICIO SOBRE COMPLEJIDAD Y RUIDO\n')
print ('Ejercicio 1\n')

#Datos de la nube de puntos con simula_unif
datos_unif = simula_unif(50, 2, [-50, 50])
#Datos de la nube de puntos con simula_gaus
datos_gauss = simula_gaus(50, 2, [5, 7])

#Muestra los datos
print ('Gráfica con nube de puntos uniforme')
muestra_datos(datos_unif, 'Nube de puntos uniforme')
print ('Gráfica con nube de puntos siguiendo una normal')
muestra_datos(datos_gauss, 'Nube de puntos normal')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

#a)---------------------------------------------------------

print ('Ejercicio 2\n')

#Función que asigna etiquetas
def f(a, b, x1, x2):
        return np.sign(x2 - a*x1 - b)

#Asigno la etiqueta segun el signo de la funcion f(x,y) = y-ax-b
def asigno_etiquetas(datos, a, b):

        #Vector con etiquetas de los datos
        im_datos = []

        #Meto las etiquetas en el vector im_datos
        for i in range(0, len(datos)):
                im_datos.append(f(a, b, datos[i][0], datos[i][1]))

        return datos, im_datos


#Dibuja los datos de la nube de puntos
def grafica(datos, im_datos, a, b, titulo):
        pos_x = [] #Coordenada X de los datos con etiqueta 1
        pos_y = [] #Coordenada Y de los datos con etiqueta 1
        neg_x = [] #Coordenada X de los datos con etiqueta -1
        neg_y = [] #Coordenada Y de los datos con etiqueta -1

        #Relleno los vectores de coordenadas
        for i in range(len(datos)):
                if im_datos[i] == 1:
                        #Vector de datos con etiquetas positivas
                        pos_x.append(datos[i][0])
                        pos_y.append(datos[i][1])
                else:
                        #Vector de datos con etiquetas positivas
                        neg_x.append(datos[i][0])
                        neg_y.append(datos[i][1])
        
        #Representamos los datos
        plt.scatter(pos_x, pos_y, c='red', label = 'Puntos con etiqueta positiva')
        plt.scatter(neg_x, neg_y, c='b', label = 'Puntos con etiqueta negativa')

        #print('\n GRAFICA \n')
        #print('len(datos)' + str(len(datos)))
        #print('len(pos)' + str(len(pos_x)))
        #print('len(neg)' + str(len(neg_x)))

        X = np.linspace(-60, 60, 100, endpoint=True)
        Y = a*X+b
        plt.plot(X, Y, color="black")

        #Añado el título 
        plt.title(titulo)
        #Añado la leyenda
        plt.legend(loc=2)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()

#Calculo los parametros de la recta
a, b = simula_recta([-50, 50])
#Devuelvo los datos con sus etiquetas segun la función signo(y-ax-b)
datos_unif, im_datos = asigno_etiquetas(datos_unif, a, b)
#Dibujo la gráfica
print('Gráfica con datos etiquetados sin ruido')
grafica(datos_unif, im_datos, a, b, 'Datos con sus etiquetas (sin ruido)')

#b)---------------------------------------------------------

#Asigno la etiqueta con ruido
def asigno_etiquetas_ruido(datos, a, b):

        #Vector con etiquetas de los datos
        im_datos = []
        pos = 0 #Numero datos con etiquita positiva
        neg = 0 #Numero datos con etiquita negativa

        #Miro el numero de datos positivos y negativos que hay
        for i in range(0, len(datos)):
                if f(a, b, datos[i][0], datos[i][1]) == 1:
                        pos = pos+1
                else:
                        neg = neg+1

        #Numero de datos con ruido: 10% 
        ruido_pos = int(0.1*pos)
        ruido_neg = int(0.1*neg)

        #print('\n ruido \n')
        #print('len(datos)' + str(len(datos)))
        #print('len(pos)' + str(pos))
        #print('len(neg)' + str(neg))

        #Meto las etiquetas con ruido en el vector im_datos
        for i in range(0, len(datos)):
                if f(a, b, datos[i][0], datos[i][1]) == 1:
                        if ruido_pos > 0:
                                im_datos.append(-f(a, b, datos[i][0], datos[i][1]))
                                ruido_pos = ruido_pos-1
                                #print('ruido_pos' + str(ruido_pos))
                        else:
                                im_datos.append(f(a, b, datos[i][0], datos[i][1]))
                else:
                        if ruido_neg > 0:
                                im_datos.append(-f(a, b, datos[i][0], datos[i][1]))
                                ruido_neg = ruido_neg-1
                                #print('ruido_neg' + str(ruido_neg))
                        else:
                                im_datos.append(f(a, b, datos[i][0], datos[i][1]))

        return datos, im_datos


#Devuelvo los datos y sus etiquetas con ruido
datos_unif, im_datos = asigno_etiquetas_ruido(datos_unif, a, b)
#Dibujo la gráfica
print('Gráfica con datos etiquetados con ruido')
grafica(datos_unif, im_datos, a, b, 'Datos con sus etiquetas (con ruido)')


input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 3 -------------------------------------#

print ('Ejercicio 3\n')

#Función frontera del clasificador
def f1(x, y):
        return (x-10)**2 + (y-20)**2 - 400

#Función frontera del clasificador
def f2(x, y):
        return 0.5*(x+10)**2 + (y-20)**2 - 400

#Función frontera del clasificador
def f3(x, y):
        return 0.5*(x-10)**2 - (y+20)**2 - 400

#Función frontera del clasificador
def f4(x, y):
        return y - 20*x**2 - 5*x +3

