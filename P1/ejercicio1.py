# -*- coding: utf-8 -*-

#############################
#####     LIBRERIAS     #####
#############################
import numpy as np
from sympy import *
import math as m
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------#
#------------- Ejercicio sobre la búsqueda iterativa de óptimos ----------------#
#-------------------------------------------------------------------------------#

#------------------------------Ejercicio 1 -------------------------------------#

# Fijamos la semilla

def E(w): 
        u, v = w[0], w[1]
        return (u*m.exp(v)-2*v*m.exp(-u))**2

# Derivada parcial de E respecto de u
def Eu(w):
        u, v = w[0], w[1]
        return 2*(u*m.exp(v)-2*v*m.exp(-u))*(m.exp(v)+2*v*m.exp(-u))

# Derivada parcial de E respecto de u
def Ev(w):
        u, v = w[0], w[1]
        return 2*(u*m.exp(v)-2*v*m.exp(-u))*(u*m.exp(v)-2*m.exp(-u))
		
# Gradiente de E
def gradE(w):
	return np.array([Eu(w), Ev(w)])

#lr: learning rate, grad_fun: gradE,
#epsilon: valor de E(u, v) que queremos alcanzar

#Función que realiza el algoritmo del gradiente descendente hasta obtener un w tal que f(w)<epsilon
def gd(w, lr, grad_fun, fun, epsilon, max_iters):
        for it in range(0, max_iters):
                if fun(w)<epsilon:
                        break
                else:
                        w = w - lr*grad_fun(w)
                        #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))
                        
        return it, w

print ('\nGRADIENTE DESCENDENTE')
print ('\nEjercicio 1\n')

w = np.array([1,1]) #w: punto inicial  
num_ite, w = gd(w, 0.1, gradE, E, 10**(-14), 100) #Ejecutamos el algoritmo de gradiente descendente

#b)Cuántas iteraciones tarda el algoritmo en obtener por primera vez un valor de E(u, v) inferior a 10^-14

print ('Numero de iteraciones: ', num_ite)
input("\n--- Pulsar tecla para continuar ---\n")

#c)¿En qué coordenadas (u, v) se alcanzó por primera vez un valor igual o menor a 10^−14 en el apartado anterior.
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

input("\n--- Pulsar tecla para continuar ---\n")


#------------------------------Ejercicio 2 -------------------------------------#

#Función a minimizar
def f(w):
        x, y = w[0], w[1]
        return (x-2)**2 + 2*(y+2)**2 + 2*m.sin(2*np.pi*x)*m.sin(2*np.pi*y)
	
# Derivada parcial de f respecto de x
def fx(w):
        x, y = w[0], w[1]
        return 2*(x-2) + 2*m.sin(2*np.pi*y)*m.cos(2*np.pi*x)*2*np.pi

# Derivada parcial de f respecto de y
def fy(w):
        x, y = w[0], w[1]
        return 4*(y+2) + 2*m.sin(2*np.pi*x)*m.cos(2*np.pi*y)*2*np.pi
	
# Gradiente de f
def gradf(w):
        return np.array([fx(w), fy(w)])
	
# a) Usar gradiente descendente para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1

#Funcion que ejecuta el algoritmo de gradiente descendente y dibuja la gráfica
def gd_grafica(w, lr, grad_fun, fun, max_iters):
        fw_record = [] #Guardamos los valores de f(w) de todos los w calculados por el algoritmo

        #Ejecución del algoritmo
        for it in range(0, max_iters):
                w = w - lr*grad_fun(w)
                fw_record.append(fun(w))
                #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))
                        
        #Dibujando el resultado
        plt.plot(range(0,max_iters), fw_record, "bo")
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x,y)')
        plt.show()
        return w

print ('Resultados ejercicio 2\n')

w = np.array([1,-1]) #Punto inicial

print ('\nGrafica con learning rate igual a 0.01')
gd_grafica(w, 0.01, gradf, f, 50)
print ('\nGrafica con learning rate igual a 0.1')
gd_grafica(w, 0.1, gradf, f, 50)
input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

#Algoritmo de gradiente descendente
def gd(w, lr, grad_fun, fun, max_iters):
        for it in range(0, max_iters):
                w = w - lr*grad_fun(w)
                #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))         
        return w

print ('Punto de inicio: (2.1, -2.1)\n')

w = np.array([2.1,-2.1])
w = gd(w, 0.01, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')

w = np.array([3.0,-3.0])
w = gd(w, 0.01, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')

w = np.array([1.5,1.5])
w = gd(w, 0.01, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')

w = np.array([1.0,-1.0])
w = gd(w, 0.01, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")
