#Harrison Denning Jan 17, 2024
#this is code for HW 1 prblm 2 of Optomization Techniques
#This code finds the minimal travel time along a path from (0,1) to (1,0) including friction
import time #import this to use tme.time to find wall clock time
import numpy as np # stands for numerical python, gives numerical methods tools great for matrix algebra
from scipy.optimize import minimize # calling in a specific function (minimize) from optomize module of scipy
import matplotlib.pyplot as plt

start = time.time()
n=10 # there are n-2 design variables, n nodes
j = 0 # a counter variable
y_constraints = (1,0)
y_val = np.linspace(y_constraints[0],y_constraints[1],n) # make an array with n nodes between your two y constraints

y_to_pass = y_val[1:-1] # make an array to pass to the function with just the middle values


def f(y):
    T= 0
    global j
    j += 1
    m = 0.3
    h = 1 # initial starting height
    grav_const = np.sqrt(2/9.81) # add in the grav const, use 9.81 on bottom for earth
    x = np.linspace(0,1,n) # these are the x values which will be used in the function
    y = np.append(y, [0]) #add 1 and 0 back on to the y values now
    y = np.concatenate(([1], y))
    for i in range(len(x)-1):
        T += grav_const * ((np.sqrt((x[i + 1] - x[i])**2 + (y[i + 1] - y[i])**2))/(np.sqrt(h - y[i + 1] - m * x[i + 1]) + np.sqrt(h - y[i] - m * x[i])))
    return T

#perform the optomization and store the results in arrays ready to be graphed
res = minimize(f, y_to_pass)
y_results = res.x
y_results = np.append(y_results, [0])
y_results = np.concatenate(([1], y_results))
x_graph = np.linspace(0,1,n)  
print(j)

# end time
end = time.time()

# # total time taken
print("Execution time of the program is- ", end-start)

plt.figure()
plt.plot( x_graph, y_results)
# plt.scatter(x_graph,y_results)
plt.title('Brachistochrome with n=12 Nodes')
plt.xlabel('Horizontal Ditance')
plt.ylabel('Vertical Distance')
plt.show()

# print(res)