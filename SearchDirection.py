# Harrison Denning 25 Jan 2024
# line search algorithm

import numpy as np #stands for numerical python, gives numerical methods tools great for matrix algebra
from scipy.optimize import minimize 
import matplotlib.pyplot as plt

k = 0 #global var to count func call

# first define f,  phi (function of x, alpha and p)
def practice(x):
    global k
    k = k + 1
    return x[0]**2 + x[1]**2
def practice_prime(x):
    return np.array([2*x[0], 2*x[1]])

def SQ(x): # slanted quadratic function
    global k
    k = k + 1
    return x[0]**2+x[1]**2 - 1.5*x[0]*x[1] #this is an arbitrary function, put in what you want to sovlve

def SQ_prime(x): #Slanted quadratic gradient
    return np.array([2*x[0]-1.5*x[1], 2*x[1]-1.5*x[0]]) #for these easy functions just do the derivative by hand

def Rosenbrock(x):
    global k
    k = k + 1
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
def Rosenbrock_prime(x):
    return np.array([-400*x[0]*(x[1]-x[0]**2)-2*(1-x[0]), 200*(x[1]-x[0]**2)])

def Jones(x):
    global k
    k = k + 1
    return x[0]**4 + x[1]**4 - 4*x[0]**3 - 3*x[1]**3 + 2*x[0]**2 + 2*x[0]*x[1]

def Jones_prime(x):
    return np.array([4*x[0]**3 - 12*x[0]**2 + 4*x[0] + 2*x[1], 4*x[1]**3 - 9*x[1]**2 + 2*x[0]])

#phi and phi prime functions defined for a given p and x
def phi(f, x, alpha, p):
    return f(x + alpha*p)


def phi_prime(f_prime, x, alpha, p):
    return f_prime(x + alpha*p)@p

# bracketing:
def Bracket(func, func_prime, x0, p, meu1, meu2, alpha_init, step_inc): #, phi0, phi0_prime):
    phi0 = phi(func, x0, 0, p)
    phi0_prime = phi_prime(func_prime, x0, 0, p)
    # print(phi0, 'is phi0', phi0_prime, 'is phi0_prime')
    alpha1 = 0
    alpha2 = alpha_init
    phi1 = phi0
    phi1_prime = phi0_prime
    first = True
    alpha_min = 0 #create a var to store the min value in

# decide what your sufficient decrease (meu1) and curvature (meu2) factors are
# determine by how much to increase your alpha each iteration

# until you find the a place to stop
    while True:
        
        phi2 = phi(func,x0, alpha2, p)
        # print('alpha 2:', alpha2)
        # print('if', phi2, '>',phi0 + meu1*alpha2*phi0_prime)
        # print('phi0: ', phi0, 'meu1: ', meu1, 'alpha2: ', alpha2, 'phi0_prime: ', phi0_prime)
# check what phi is equal to once you move alpha along the line
        if (phi2 > phi0 + meu1*alpha2*phi0_prime) or (not first and phi2 > phi1):
            # print('above line of sufficient decrease', alpha1, alpha2)
            alpha_min = Pinpoint(func, func_prime, x0, p, alpha1, alpha2, meu1, meu2, phi0, phi0_prime)
            return alpha_min
        phi2_prime = phi_prime(func_prime, x0, alpha2, p)
# if your phi(alpha_First step) lies above the line of sufficient decrease or if its not your first initial apha guess and you're below the line of sufficient decrease
    #then go ahead and pinpoint()
        if abs(phi2_prime) <= -meu2*phi0_prime:
            # print('not bracketed, just min')
            alpha_min = alpha2
            return alpha_min
        elif phi2_prime >= 0:
            # print('Bracketed between2:', alpha2, alpha1)
            alpha_min = Pinpoint(func, func_prime, x0, p, alpha2, alpha1, meu1, meu2, phi0, phi0_prime)
            return alpha_min
        else:
            # print(alpha2, 'is now alpha 1, alpha2 = ', alpha2*step_inc)
            alpha1 = alpha2
            alpha2 = alpha2*step_inc
        first = False

    # if absval(phi2') is within the acceptle curvature then accept that phi2 as the minimum point
    # else if slope is + pinpoint
    # else increment your alhpa step

# Pinpointing if you've gone through bracketing and not found it
def Pinpoint(func, func_prime, x0, p, alpha_low, alpha_high, meu1, meu2, phi0, phi0_prime):
    the_min = 0
    # print('p_dir = ', p)
    i = 0 # count how many times the function runs
    # print(alpha_high, alpha_low)
    # Until you find the minimu point do the following
    while True:
        # take a guess alpha somewhere between the brackets you found (bisecting?)
        alpha_p = (alpha_high+alpha_low)/2
        phip = phi(func, x0, alpha_p, p)
        #between the two bracketed points, get your test point
        # interpolation would be better, but we'll do bisection for now
        if i > 2000:
            print('kill switch')
            return alpha_p
# if that guess is above the line of sufficient decrease OR above your lowest bracketing point
        elif phip > phi0 + meu1*alpha_p*phi0_prime or phip > phi(func,x0, alpha_low, p):
            alpha_high = alpha_p # then that guess alpha is your new high-end bracekt point
            # print()
            # print('above LSD')
        
# else, if your guess alpha' is less than or eqaul to your min curvature, you found the point! (return it)
        else:
            # print('its else')
            phip_prime = phi_prime(func_prime, x0, alpha_p, p)
            # print('This is the condition for elif now', phi_prime(func_prime, x0, alpha_high-alpha_low, p))
            if abs(phip_prime) <= -meu2*phi0_prime:
                # print('its if')
                the_min = alpha_p
                return alpha_p
            elif phi_prime(func_prime, x0, alpha_high-alpha_low, p) >= 0:
                # print('elif was used')
                alpha_high = alpha_low
            
            alpha_low = alpha_p
        i = i + 1
        
    # else if your alpa guess' * (alpha_high - Alpha_low) 
    # make your alhpa_guess the new low bracket

def SteepestDecent(func, func_prime, x0, alpha_k, m1, m2, step): #pass in a starting point, and a convergence tolerance
    # setup initial var
    global j
    j = 0
    global values, w
    w = []
    values = []
    x_new = x0
    while (np.linalg.norm(func_prime(x0),ord=np.inf) > .01) :
        x0 = x_new
        p_dir = - func_prime(x0) / np.linalg.norm(func_prime(x0))
        # print('p_dir: ', p_dir)
        alpha_pass = alpha_k
        alpha_k = Bracket(func, func_prime, x0, p_dir, m1, m2, alpha_pass, step) #, phi0, phi0_prime)
        x_new = x0 + alpha_k*p_dir
        
        x_values = [x0[0], x_new[0]]
        y_values = [x0[1], x_new[1]]
        plt.plot(x_values, y_values, 'bo', linestyle="--")
        values.append(np.linalg.norm(func_prime(x0)))
        w.append(j)
        j += 1

    return x_new #return an optimal point x* and f(x*) minimum function value
    
def ConjugateGrad(func, func_prime, x0, alpha, m1, m2, step):
    global j
    j = 0 # a counter var
    global values, w
    w = []
    values = []
    x_new = x0
    alpha_k = alpha
    while (np.linalg.norm(func_prime(x0),ord=np.inf) > .01):
        alpha_pass = alpha_k
        reset = np.abs(np.dot(np.transpose(func_prime(x_new)), func_prime(x0)))/np.abs(np.dot(np.transpose(func_prime(x0)), func_prime(x0)))
        # print('this is reset value', reset)
        if k == 0 or reset >= .1:
            p_dir = - func_prime(x0) / np.linalg.norm(func_prime(x0))
        else:
            # print('conjugated the gradient')
            beta_k = np.dot(np.transpose(func_prime(x_new)), func_prime(x_new))/np.dot(np.transpose(func_prime(x0)), func_prime(x0))
            p_dir = - func_prime(x0) / np.linalg.norm(func_prime(x0)) + beta_k*p_dir_old
        # print('P_dir', p_dir)
        p_dir_old = p_dir
        x0 = x_new
        alpha_k = Bracket(func, func_prime, x0, p_dir, m1, m2, alpha_pass, step) 
        x_new = x0 + alpha_k*p_dir
        
        x_values = [x0[0], x_new[0]]
        y_values = [x0[1], x_new[1]]
        plt.plot(x_values, y_values, 'bo', linestyle="--")
        
        values.append(np.linalg.norm(func_prime(x0)))
        w.append(j)
        j += 1
    return x_new

def QuasiNewton(func, func_prime, x0, alpha, m1, m2, step):
    global j
    j = 0
    global values, w
    w = []
    values = []
    x_new = x0
    I = np.identity(func_prime(x0).size)
    alpha_k = alpha
    # print('this is inverse norm value: ',(1/np.linalg.norm(func_prime(x0))))
    while (np.linalg.norm(func_prime(x0),ord=np.inf) > .01):
        reset = np.abs(np.dot(np.transpose(func_prime(x_new)), func_prime(x0)))/np.abs(np.dot(np.transpose(func_prime(x0)), func_prime(x0)))
        # print("while 0.5 < ", np.linalg.norm(func_prime(x0),ord=np.inf))
        if k == 0 or reset >= .1:
            V_k = (1/np.linalg.norm(func_prime(x0)))*I
            # print('this is V_k',V_k)
        else:
            s = x_new - x0
            y = func_prime(x_new) - func_prime(x0)
            sigma = 1/(np.transpose(s)@y)
            V_k = (I-sigma*s@(np.transpose(y)))@V_k_old@(I-sigma*(y@np.transpose(s))) + sigma*(s@np.transpose(s))

        V_k_old = V_k
        p_dir = -V_k@func_prime(x0)
        # print('p_dir: ', p_dir)
        x0 =x_new
        # print('This is being passed in x0:', x0, 'p_dir:', p_dir, 'm1: ', m1, "m2", m2, 'alpha_k: ', alpha_k, 'step: ', step)
        alpha_k = Bracket(func, func_prime, x0, p_dir, m1, m2, alpha_k, step) 
        
        x_new = x0 + alpha_k*p_dir
        # print('x_new', x_new)
    
        x_values = [x0[0], x_new[0]]
        y_values = [x0[1], x_new[1]]
        plt.plot(x_values, y_values, 'bo', linestyle="--")
        values.append(np.linalg.norm(func_prime(x0)))
        w.append(j)
        j += 1
    return x_new

# p_dir = np.array([-1,1]) # for sq -1,1 for Rb 1,-3 for Jns 1,2 
#these are the functions you may chose to put in, remeber to chagne p and x0 up above!
pr = practice
pr_prime = practice_prime
sq = SQ
sq_prime = SQ_prime
Rb = Rosenbrock
Rb_prime = Rosenbrock_prime
Jns = Jones
Jns_prime = Jones_prime
#change these for your function you're using

x0 = np.array([1,1]) # for sq 2,-6 for Rb 0,2 for Jns 1,1 first point convergnce. -1.73003408, -1.69611452
alpha_k = 1
m1 = 0.001
m2 = 0.1
step = 1.5
# x0 = SteepestDecent(Jns, Jns_prime, x0, alpha_k, m1, m2, step)
x0 = QuasiNewton(Jns, Jns_prime, x0, alpha_k, m1, m2, step)
res = minimize(Jns, x0)
print('best point is', x0)
print(res)
print('iterations: ', j) #j for SD, k for CGrad, i for Quasi
print('Function calls: ', k)
n1 = 100
x1 = np.linspace(-3,3,n1) # use numpy linspace (linear space) to give values between -5 to 5 and 100 of them(all equallyspaced)
n2 = 200
x2 = np.linspace(-3,3,n2)

fun_output = np.zeros([n1,n2]) #create a matrix n1 by n2 populated with zeroes
for i in range(n1): #range will go to 100th element of n1 (which is 99)
    for j in range(n2):
        fun_output[i,j] = Jns([x1[i], x2[j]])
        
plt.contour(x1,x2,np.transpose(fun_output), 100, linewidths = 2, alpha =0.9, zorder = 2 ) #range of plot and the output you want to plot, don't forget to avoid transpose errors
plt.colorbar()
plt.plot(x0[0], x0[1], 'r')
plt.figure()
# plt.yscale("log")
# plt.axis([0, 10, 0,5])
plt.plot(values, w, 'o')
plt.show()
print(values)
print(w)