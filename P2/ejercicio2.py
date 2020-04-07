# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

#------------------ Funciones de la sección anterior -------------------------------------#

# Fijamos la semilla
np.random.seed(1)

#Devuelve N vectores de dimensión dim dentro del rango
def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

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

#------------------------------Ejercicio 1 -------------------------------------#

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

    return w, iters


#Crea un vector con el formato de datos correcto
def formato_datos(datos):
        x=[]# Vector con datos en el formato adecuado
        
        #Añadimos los datos en el formato adecuado al vector x
        for i in range(0, len(datos)):
                x.append(np.array([1, datos[i][0], datos[i][1]]))
        x = np.array(x, np.float64)
        return x

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
                        pos_x.append(datos[i][0])
                        pos_y.append(datos[i][1])
                else:
                        #Vector de datos con etiquetas positivas
                        neg_x.append(datos[i][0])
                        neg_y.append(datos[i][1])
        
        #Representamos los datos
        plt.scatter(pos_x, pos_y, c='red', label = 'Puntos con etiqueta positiva')
        plt.scatter(neg_x, neg_y, c='b', label = 'Puntos con etiqueta negativa')

        X = np.linspace(-60, 60, 100, endpoint=True)
        Y = -(w[0] + w[1]*X)/w[2]
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


print ('MODELOS LINEALES\n')
print ('Ejercicio 1\n')

#a)---------------------------------------------------------

#Datos de la nube de puntos con simula_unif
datos_unif = simula_unif(50, 2, [-50, 50])
#Calculo los parametros de la recta
a, b = simula_recta([-50, 50])
#Devuelvo los datos y sus etiquetas sin ruido
datos_unif, im_datos = asigno_etiquetas(datos_unif, a, b)
#Pongo los datos en el formato correcto para el algoritmo PLA
datos = formato_datos(datos_unif)

print('\n ETIQUETAS SIN RUIDO \n')

#Inicializo con vector 0
w = np.zeros(3, np.float64)
w, iters = ajusta_PLA(datos, im_datos, 100, w)
print('Vector inicial: vector cero')
print('Los pesos obtenidos son: ' + str(w))
print('Fueron necesarias ' + str(iters) + ' iteraciones para converger.')

#Grafica con los resultados obtenidos
grafica(datos_unif, im_datos, w, 'PLA con vini = 0')

#Inicializo con vector aleatorio
for i in range(0, 10):
    sum_iters = 0 #Suma de las iteraciones necesarias para converger
    
    #Inicializo el vector aleatorio
    w = []
    for i in range(0,3):
        #Le meto 3 valores aleatorios ente 0 y 1
        w.append(random.random())
    #Convierto w en un array de numpy
    w = np.array(w, np.float64)
    
    #Ejecuto el PLA
    w, iters = ajusta_PLA(datos, im_datos, 100, w)
    sum_iters += iters

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('\nVector inicial: vector aleatorio')
print('Los pesos obtenidos en la ultima iteración son: ' + str(w))
print('La media de las iteraciones necesarias para converger es: ' + str(iters))

#Grafica con los resultados obtenidos en la última iteración
grafica(datos_unif, im_datos, w, 'PLA con vini aleatorio')

input("\n--- Pulsar tecla para continuar ---\n")

#b)---------------------------------------------------------

print('\n ETIQUETAS CON RUIDO \n')

#Devuelvo los datos y sus etiquetas con ruido
datos_unif, im_datos = asigno_etiquetas_ruido(datos_unif, a, b)
#Pongo los datos en el formato correcto para el algoritmo PLA
datos = formato_datos(datos_unif)

#Inicializo con vector 0
w = np.zeros(3, np.float64)
w, iters = ajusta_PLA(datos, im_datos, 100, w)
print('Vector inicial: vector cero')
print('Los pesos obtenidos son: ' + str(w))
print('Fueron necesarias ' + str(iters) + ' iteraciones para converger.')

#Grafica con los resultados obtenidos
grafica(datos_unif, im_datos, w, 'PLA con vini = 0')

#Inicializo con vector aleatorio
for i in range(0, 10):
    sum_iters = 0 #Suma de las iteraciones necesarias para converger
    
    #Inicializo el vector aleatorio
    w = []
    for i in range(0,3):
        #Le meto 3 valores aleatorios ente 0 y 1
        w.append(random.random())
    #Convierto w en un array de numpy
    w = np.array(w, np.float64)
    
    #Ejecuto el PLA
    w, iters = ajusta_PLA(datos, im_datos, 100, w)
    sum_iters += iters

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('\nVector inicial: vector aleatorio')
print('Los pesos obtenidos en la ultima iteración son: ' + str(w))
print('La media de las iteraciones necesarias para converger es: ' + str(iters))

#Grafica con los resultados obtenidos en la última iteración
grafica(datos_unif, im_datos, w, 'PLA con vini aleatorio')

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#
#Asigno la etiqueta segun el signo de la funcion f(x,y) = y-ax-b
def asigno_etiquetas_prob(datos, a, b):

        #Vector con etiquetas de los datos
        im_datos = []

        #Meto las etiquetas en el vector im_datos
        for i in range(0, len(datos)):
           if f(a, b, datos[i][0], datos[i][1]) != 1:
               im_datos.append(0)
           else:
               im_datos.append(1)

        return datos, im_datos

def




#Datos de la nube de puntos con simula_unif
datos_unif = simula_unif(100, 2, [0, 2])
#Calculo los parametros de la recta que hace de frontera
a, b = simula_recta([0, 2])
#Devuelvo los datos y sus etiquetas (0/1)
datos_unif, im_datos = asigno_etiquetas_prob(datos_unif, a, b)
