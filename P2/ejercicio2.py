# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
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
def grafica(datos, im_datos, w, titulo, rango, var):
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

        X = np.linspace(rango[0], rango[1], 100, endpoint=True)
        Y = -(w[0] + w[1]*X)/w[2]
        plt.plot(X, Y, color="black")

        #Fijo los ejes
        axes = plt.gca()
        axes.set_xlim([rango[0]-var, rango[1]+var])
        axes.set_ylim([rango[0]-var, rango[1]+var])
        #Añado el título 
        plt.title(titulo)
        #Añado la leyenda
        plt.legend(loc=4)
        #Ponemos nombre a los ejes
        plt.xlabel('Coordenada x')
        plt.ylabel('Coordenada y')
        #Pintamos la gráfica
        plt.show()


print ('MODELOS LINEALES\n')
print ('Ejercicio 1\n')

#a)---------------------------------------------------------

# Fijamos la semilla
np.random.seed(1)

#Datos de la nube de puntos con simula_unif
datos_unif = simula_unif(100, 2, [-50, 50])
#Calculo los parametros de la recta
a, b = simula_recta([-50, 50])
#Devuelvo los datos y sus etiquetas sin ruido
datos_unif, im_datos = asigno_etiquetas(datos_unif, a, b)
#Pongo los datos en el formato correcto para el algoritmo PLA
datos = formato_datos(datos_unif)

print('\n ETIQUETAS SIN RUIDO \n')

#Inicializo con vector 0
print('Vector inicial: vector cero')

sum_iters = 0 #Suma de las iteraciones necesarias para converger

for i in range(0,10):
        w = np.zeros(3, np.float64)
        w, iters = ajusta_PLA(datos, im_datos, 500, w)
        sum_iters += iters
        
        #print('Los pesos obtenidos son: ' + str(w))
        #print('Fueron necesarias ' + str(iters) + ' iteraciones para converger.')

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('La media de las iteraciones necesarias para converger es: ' + str(med_iters))
print('Los pesos son: ' + str(w))

#Grafica con los resultados obtenidos
grafica(datos_unif, im_datos, w, 'PLA con vini = 0', [-50, 50], 10)

#Inicializo con vector aleatorio
print('\nVector inicial: vector aleatorio')

sum_iters = 0 #Suma de las iteraciones necesarias para converger

for i in range(0, 10):
    
    #Inicializo el vector aleatorio
    w = []
    for i in range(0,3):
        #Le meto 3 valores aleatorios ente 0 y 1
        w.append(np.random.random())
    #Convierto w en un array de numpy
    w = np.array(w, np.float64)
    
    #Ejecuto el PLA
    w, iters = ajusta_PLA(datos, im_datos, 500, w)
    sum_iters += iters

    #print('Los pesos obtenidos son: ' + str(w))
    print('Las iteraciones necesarias para converger son: ' + str(iters))

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('La media de las iteraciones necesarias para converger es: ' + str(med_iters))
print('Los pesos obtenidos en la última ejecución son: ' + str(w))

#Grafica con los resultados obtenidos en la última iteración
grafica(datos_unif, im_datos, w, 'PLA con vini aleatorio', [-50, 50], 10)

input("\n--- Pulsar tecla para continuar ---\n")

#b)---------------------------------------------------------

print('\n ETIQUETAS CON RUIDO \n')

#Devuelvo los datos y sus etiquetas con ruido
datos_unif, im_datos = asigno_etiquetas_ruido(datos_unif, a, b)
#Pongo los datos en el formato correcto para el algoritmo PLA
datos = formato_datos(datos_unif)

#Inicializo con vector 0

print('Vector inicial: vector cero')

sum_iters = 0 #Suma de las iteraciones necesarias para converger

for i in range(0,10):
        w = np.zeros(3, np.float64)
        w, iters = ajusta_PLA(datos, im_datos, 500, w)
        sum_iters += iters

        #print('Los pesos obtenidos son: ' + str(w))
        #print('Fueron necesarias ' + str(iters) + ' iteraciones para converger.')

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('La media de las iteraciones necesarias para converger es: ' + str(med_iters))
print('Los pesos son: ' + str(w))

#Grafica con los resultados obtenidos
grafica(datos_unif, im_datos, w, 'PLA con vini = 0', [-50, 50], 10)

#Inicializo con vector aleatorio
print('\nVector inicial: vector aleatorio')

sum_iters = 0 #Suma de las iteraciones necesarias para converger

for i in range(0, 10):    
    #Inicializo el vector aleatorio
    w = []
    for i in range(0,3):
        #Le meto 3 valores aleatorios ente 0 y 1
        w.append(np.random.random())
    #Convierto w en un array de numpy
    w = np.array(w, np.float64)
    
    #Ejecuto el PLA
    w, iters = ajusta_PLA(datos, im_datos, 500, w)
    sum_iters += iters

    #print('Los pesos obtenidos son: ' + str(w))
    #print('Las iteraciones necesarias para converger son: ' + str(iters))

#Media de las iteraciones necesarias para converger
med_iters = sum_iters/10

print('La media de las iteraciones necesarias para converger es: ' + str(med_iters))
print('Los pesos obtenidos en la última ejecución son: ' + str(w))

#Grafica con los resultados obtenidos en la última iteración
grafica(datos_unif, im_datos, w, 'PLA con vini aleatorio', [-50, 50], 10)

input("\n--- Pulsar tecla para continuar ---\n")

#------------------------------Ejercicio 2 -------------------------------------#

print ('Ejercicio 2\n')

#a)-------------------------------------------------

#Función sigmoide
def sigma(x):
    return np.exp(x)/(np.exp(x)+1)


# Regresion Logistica usando Gradiente Descendente Estocastico
def rl_sgd(x, y, lr, max_iters):

        #Inicializa el punto inicial a (0,0)
        w = np.array([])
        for i in range(0, len(x[0])):
                w = np.append(w, 0.0)

        w = np.array(w, np.float64)
        #Inicializo w_old a un valor con norma mayor que 0.01
        w_old = np.array([1,1,1])

        iters = 0
        #Paro cuando ||w^t - w^(t-1)||<0.01
        while np.linalg.norm(w-w_old)>=0.01 and iters < max_iters:

                #Actualizo el w_old
                w_old = w
                
                #Barajo los datos del vector x y las etiquetas en el mismo orden
                c = list(zip(x, y))
                np.random.shuffle(c)
                x, y = zip(*c)

                #Algoritmo de Gradiente descendente
                for i in range(0, len(x)):
                        #Actualiazción de w según el algoritmo
                        w = w + lr*y[i]*x[i]*sigma(-np.dot(w, x[i])*y[i])

                #print('Epoca:' + str(iters) + ' w: ' + str(w))
                #print('Sigma: ' + str(-np.dot(w, x[i])*y[i]))
                #print('Variación: ' + str(lr*y[i]*x[i]*sigma(-np.dot(w, x[i])*y[i])))
                        
                iters += 1
                        
        return w, iters

# Fijamos la semilla
np.random.seed(4)

#Datos de la nube de puntos con simula_unif
datos_unif = simula_unif(100, 2, [0, 2])
#Calculo los parametros de la recta que hace de frontera
a, b = simula_recta([0, 2])
print('Los coeficientes de la recta real son:')
print('a: ' + str(a) + '    b: ' + str(b))
#Devuelvo los datos y sus etiquetas (0/1)
datos_unif, im_datos = asigno_etiquetas(datos_unif, a, b)
#Pongo los datos en el formato correcto para el algoritmo SGD
datos = formato_datos(datos_unif)
y = np.array(im_datos)

#Ejecuto la Rl con SGD
w, iters = rl_sgd(datos, y, 0.01, 1000)
print('Los pesos obtenidos son: ' + str(w))
print('Fueron necesarias ' + str(iters) + ' iteraciones.')

print('La gráfica con el resultado obtenido es:')
grafica(datos_unif, im_datos, w, 'Regresion logistica con datos training', [0, 2], 0.5)

#b)-----------------------------------------------------------------

input("\n--- Pulsar tecla para continuar ---\n")

print('La función obtenida es g(x) = sigma(np.dot(w,x))')
print('Los pesos obtenidos son: w = ' + str(w))

#Creo la muestra de datos test
datos_test = simula_unif(1000, 2, [0, 2])
datos_test, im_test = asigno_etiquetas(datos_test, a, b)
datos_testf = formato_datos(datos_test)

def Error(x, y, w):
        suma = 0
        for i in range(0, len(x)):
                suma = suma + np.log(1 + np.exp(-y[i]*np.dot(w, x[i])))
        return suma/len(x)

e_in = Error(datos, im_datos, w)
e_out = Error(datos_testf, im_test, w)
print('El error Ein es : ' + str(e_in))
print('El error Eout es : ' + str(e_out))

print('La gráfica con el resultado obtenido con los datos test es:')
grafica(datos_test, im_test, w, 'Regresion logistica con datos test', [0, 2], 0.5)
