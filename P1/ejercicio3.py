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

#------------------------------BONUS -------------------------------------#

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

# Derivada parcial de f respecto de x dos veces
def fxx(w):
        x, y = w[0], w[1]
        return 2 - 8*np.pi**2*m.sin(2*np.pi*y)*m.sin(2*np.pi*x)

# Derivada parcial de f respecto de x y luego respecto de y
def fxy(w):
        x, y = w[0], w[1]
        return 8*np.pi**2*m.cos(2*np.pi*y)*m.cos(2*np.pi*x)

# Derivada parcial de f respecto de y y luego respecto de x
def fyx(w):
        x, y = w[0], w[1]
        return 8*np.pi**2*m.cos(2*np.pi*x)*m.cos(2*np.pi*y)

# Derivada parcial de f respecto de y dos veces
def fyy(w):
        x, y = w[0], w[1]
        return 4 - 8*np.pi**2*m.sin(2*np.pi*x)*m.sin(2*np.pi*y)
	
# Gradiente de f
def gradf(w):
        return np.array([fx(w), fy(w)])

# Hessiana de f
def Hess(w):
        return np.array([[fxx(w), fxy(w)], [fyx(w), fyy(w)]])
	
# a) Usar el método de newton para minimizar la función f, con punto inicial (1,1)
# tasa de aprendizaje 0.01 y max 50 iteraciones. Repetir con tasa de aprend. 0.1

#Funcion que ejecuta el método Newton y dibuja la gráfica
def Newton_grafica(w, hess, grad_fun, fun, max_iters):
        fw_record = [] #Guardamos los valores de f(w) de todos los w calculados por el algoritmo

        #Ejecución del algoritmo
        for it in range(0, max_iters):
                w = w - np.dot(np.linalg.inv(hess(w)), grad_fun(w))
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

print ('\nGrafica del método de Newton')
Newton_grafica(w, Hess, gradf, f, 50)

input("\n--- Pulsar tecla para continuar ---\n")


# b) Obtener el valor minimo y los valores de (x,y) con los
# puntos de inicio siguientes:

#Algoritmo de gradiente descendente
def newton(w, hess, grad_fun, fun, max_iters):
        for it in range(0, max_iters):
                w = w - np.dot(np.linalg.inv(hess(w)), grad_fun(w))
                #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))         
        return w

print ('Punto de inicio: (2.1, -2.1)\n')

w = np.array([2.1,-2.1])
w = newton(w, Hess, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')

w = np.array([3.0,-3.0])
w = newton(w, Hess, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')

w = np.array([1.5,1.5])
w = newton(w, Hess, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor minimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')

w = np.array([1.0,-1.0])
w = newton(w, Hess, gradf, f, 50)

print ('(x,y) = (', w[0], ', ', w[1],')\n')
print ('Valor mínimo: ',f(w))

input("\n--- Pulsar tecla para continuar ---\n")

#-------------------------------
#Comparamos con gradiente descendente

#Gradiente descendente que devuelve vector con valores de f(w)
def gd_g(w, lr, grad_fun, fun, max_iters):
        fw_record = [] #Guardamos los valores de f(w) de todos los w calculados por el algoritmo

        #Ejecución del algoritmo
        for it in range(0, max_iters):
                w = w - lr*grad_fun(w)
                fw_record.append(fun(w))
                #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))

        return w, fw_record

#Método de Newton que devuelve vector con valores de f(w)
def Newton_g(w, hess, grad_fun, fun, max_iters):
        fw_record = [] #Guardamos los valores de f(w) de todos los w calculados por el algoritmo

        #Ejecución del algoritmo
        for it in range(0, max_iters):
                w = w - np.dot(np.linalg.inv(hess(w)), grad_fun(w))
                fw_record.append(fun(w))
                #print("Iteración: " + str(it) + " w:" + str(w) + " f(w):" + str(fun(w)))

        return w, fw_record

def grafica(v_newton, v_gd, max_iters):
        #Dibujando el resultado
        plt.title('Comparamos Newton con Gradiente descendente')
        plt.scatter(range(0,max_iters), v_newton, c='m', label = 'Método de Newton')
        plt.scatter(range(0,max_iters), v_gd, c='c', label = 'Gradiente Descendente')
        plt.legend(loc=7)
        plt.xlabel('Iteraciones')
        plt.ylabel('f(x,y)')
        plt.show()
        

print('\nComparación con el gradiente descendente\n')

print ('Punto de inicio: (2.1, -2.1)\n')

w = np.array([2.1,-2.1])
w_n, f_n = Newton_g(w, Hess, gradf, f, 50)
w_gd, f_gd = gd_g(w, 0.01, gradf, f, 50)
grafica(f_n, f_gd, 50)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (3.0, -3.0)\n')

w = np.array([3.0,-3.0])
w_n, f_n = Newton_g(w, Hess, gradf, f, 50)
w_gd, f_gd = gd_g(w, 0.01, gradf, f, 50)
grafica(f_n, f_gd, 50)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.5, 1.5)\n')

w = np.array([1.5,1.5])
w_n, f_n = Newton_g(w, Hess, gradf, f, 50)
w_gd, f_gd = gd_g(w, 0.01, gradf, f, 50)
grafica(f_n, f_gd, 50)

input("\n--- Pulsar tecla para continuar ---\n")

print ('Punto de inicio: (1.0, -1.0)\n')

w = np.array([1.0,-1.0])
w_n, f_n = Newton_g(w, Hess, gradf, f, 50)
w_gd, f_gd = gd_g(w, 0.01, gradf, f, 50)
grafica(f_n, f_gd, 50)
