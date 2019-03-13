import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Funcion Goldstein-Price
def fun(x,y):
    sal=  (1 + ( x + y + 1 )**2 *( 19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2 ) ) *(30+(2*x-3*y)**2*(18-32*x+12*x**2+48*y-36*x*y+27*y**2))
    return sal

#derivada parcial de x de funcion Goldstein-Price
def derx(x,y):
    sal=(4*(2*x-3*y)*(4*x*(3*x-9*y-8)+3*(9*y**2+16*y+6))+(2*x-3*y)**2*(4*(3*x-9*y-8)+12*x))*((x+y+1)*2*((x+y)*(3*(x+y)-14)+19)+1)+((2*x-3*y)*2*(4*x*(3*x-9*y-8)+3*(9*y**2+16*y+6))+30)*(2*(x+y+1)*((x+y)*(3*(x+y)-14)+19)+(x+y+1)*2*(6*(x+y)-14))
    return sal

#derivada parcial de y de funcion Goldstein-Price
def dery(x,y):
    sal=((y+x+1)**2*(3*y**2+6*x*y-14*y+3*x**2-14*x+19)+1)*((2*x-3*y)**2*(54*y-36*x+48)-6*(2*x-3*y)*(27*y**2-36*x*y+48*y+12*x**2-32*x+18))+(2*(y+x+1)*(3*y**2+6*x*y-14*y+3*x**2-14*x+19)+(y+x+1)**2*(6*y+6*x-14))*((2*x-3*y)**2*(27*y**2-36*x*y+48*y+12*x**2-32*x+18)+30)
    return sal

#metodo decenso por gradiente
def gradient(x,y,learned):
    x2=x-learned*derx(x,y)
    y2=y-learned*dery(x,y)
    #si con el coeficiente de entrenamiento saca a x2 o y2 del dominio -2,2 reducir el coeficiente hasta que x2 y y2 queden dentro del dominio
    while x2>2 or y2>2 or x2<-2 or y2<-2:
        learned=learned/10
        x2=x-learned*derx(x,y)
        y2=y-learned*dery(x,y)
    return x2,y2

def pricegradient(log=False):
    #obtnener coordenasas x,y aleatorias
    xi=np.random.random_sample(1)*2-1
    yi=np.random.random_sample(1)*2-1

    #maximo de iteraciones
    iterations=10000

    #coordenasas iniciales
    zi=fun(xi,yi)
    ys=np.zeros(iterations)
    xs=np.zeros(iterations)
    zs=np.zeros(iterations)
    print(xi,yi,zi)
    minimo=zi
    count=0
    learn=0.0000001
    #decenso por gradiente
    for i in range(iterations):
        xi,yi=gradient(xi,yi,learn)
        ys[i]=yi
        xs[i]=xi
        zs[i]=fun(xi,yi)
        if i%100==0:
            print(xs[i],ys[i],zs[i])
        #parar si ya no hay decrecimiento
        if minimo<zs[i]+.1:
            count=count+1
            #si ya no hay mejora aumentar el coeficiente de aprendizaje hasta un maximo determinado
            if count>10 and learn<0.0001:
                learn=learn*2
                count=0
            #si despues de cierta cantidad de iteraciones ya no hay mejora deterner el ciclo
            if count>500:
                break
        else:
            count=0
        minimo=min(zs[i],minimo)
        
    zi=fun(xi,yi)
    print(np.round(xi[0],decimals=1),np.round(yi[0],decimals=1),np.round(zi[0],decimals=1))
    print('')
    if log==False:
        return np.round(xi[0],decimals=1),np.round(yi[0],decimals=1),np.round(zi[0],decimals=1)
    else:
        return xs,ys,zs,i

hx=[]
hy=[]
hz=[]


#Este programa de ejemplo de decenso por gradiente tiene 2 funcionamientos el primero hace 20 corridas
#de decenso por gradiente y hace un histograma con los minimos encontrados, el segundo hace una sola corrida en donde
#se grafica el recorrido de x,y,z para llegar a un minimo. Para habilidar alguno de las 2 partes cambiar a True el if correspondiente


#Poner en True para una corrida con graficas----------------------------------------------
if True:

    #crear superficie
    x=np.arange(-2,2,.1)
    y=np.arange(-2,2,.1)
    X, Y = np.meshgrid(x, y)
    Z=fun(X,Y)
    #reducir altura maxima para mejor visualizacion
    #Z=np.minimum(Z,5000)
    xs,ys,zs,i=pricegradient(log=True)
    #mostrar superficie de la funcion
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, Z, cmap=cm.rainbow,
                           linewidth=0, antialiased=True,vmax=10000)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    #mostrar recorrido en X y Y por separado por problemas graficos al mostrarlos juntos
    pl=plt.figure()
    plt.plot(xs[0:i], ys[0:i])
    plt.arrow(xs[i-20], ys[i-20],xs[i-5]-xs[i-20], ys[i-5]-ys[i-20],width=0.01)
    plt.xlabel('x')
    plt.ylabel('y')

    #mostrar decenso de z por iteraciones
    pz=plt.figure()

    plt.subplot(3,1,1)
    plt.plot(xs[0:i])
    plt.xlabel('iterations')
    plt.ylabel('x')
    plt.subplot(3,1,2)
    plt.plot(ys[0:i])
    plt.xlabel('iterations')
    plt.ylabel('y')
    plt.subplot(3,1,3)
    plt.plot(zs[0:i])
    plt.xlabel('iterations')
    plt.ylabel('z')

    fig.show()
    pl.show()
    pz.show()










