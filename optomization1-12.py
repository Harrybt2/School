# this is code from following dr salmon in class Janruary 12 2024
# example program to show simple optomization

#these three are almost always required in this class, add on other packages as needed
import numpy as np #stands for numerical python, gives numerical methods tools great for matrix algebra
from scipy.optimize import minimize #calling in a specific function (minimize) from optomize module of scipy
import matplotlib.pyplot as plt



def f(x):
    return (1-x[0])**2 + (1-x[1])**2 + 0.5*(2*x[1]-x[0]**2)**2 # the ** is to raise to a power
# you can scale this by an number and you will get the exact same solution (just higher)

# constraint functions
def g1(x):
    return (2*x[0]-x[1])
def g2(x):
    return -(x[0]**2 + x[1]**2 - 4) # circle of radius 2, multiply by -1 bc of something in scipy documentation of ineq constraints

x0 = [4,9]
print(f(x0))

theconstraints = {'type':'ineq','fun':g1},{'type':'ineq','fun':g2} # find these in the documentation for minimize
thebounds = ((1,2),(-3,-1)) # set the bounds as the min max of one and the min max of the other
#optomize    
res = minimize(f,x0, constraints=theconstraints, bounds = thebounds) # tell it to minimize with an innequality constraint, and boundaries
print(res)

n1 = 100
x1 = np.linspace(-5,5,n1) # use numpy linspace (linear space) to give values between -5 to 5 and 100 of them(all equallyspaced)
n2 = 200
x2 = np.linspace(-5,5,n2)

fun_output = np.zeros([n1,n2]) #create a matrix n1 by n2 populated with zeroes
g1_output = np.zeros([n1,n2])
g2_output = np.zeros([n1,n2])
for i in range(n1): #range will go to 100th element of n1 (which is 99)
    for j in range(n2):
        fun_output[i,j] = f([x1[i], x2[j]])
        g1_output[i,j] = -g1([x1[i], x2[j]])
        g2_output[i,j] = -g2([x1[i], x2[j]])
        

# print(fun_output)

plt.figure()
plt.contour(x1,x2,np.transpose(fun_output), 100, linewidths = 2) #range of plot and the output you want to plot, don't forget to avoid transpose errors
plt.plot(res.x[0], res.x[1], "r*") # plot a point with a *
plt.colorbar() #take care of where the colorbar is called
plt.contourf(x1,x2,np.transpose(g1_output), [0,1000], linewidths = 2, colors = 'r', alpha = 0.3) #countourf makes it shaded, contour gives a line
plt.contourf(x1,x2,np.transpose(g2_output), [0, 1000], linewidths = 2, colors = 'r', alpha = 0.3)
plt.show()