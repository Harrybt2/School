# Harrison Denning
#Fabonacci series for CS ALgorithms 312 Spet 10, 2024
# 
import matplotlib.pyplot as plt
import numpy as np

## here is the stupid way

def fabonacci_series(x): # call and define the function
    if x < 2:
        h = 1
        return h
    else:
        a = fabonacci_series(x-1) # this is exponential b/c you calculate each value everytime you run it
        b = fabonacci_series(x-2)
        c = fabonacci_series(x-3)
    h = a + b * c
    return h # return the desired value
f = [1]*3

g = 0
while g <= 20:
    d = fabonacci_series(g)
    print(d)
    g = g + 1

## part b 'psuedocode' 
def fabonacci_series(x):
    return int(f[x-1] + f[x-2] * f[x-3])

f = [1]*3


res = []
n = 3
end_val = 20
while n <= end_val:
    g = fabonacci_series(n)
    f.append(g)
    n = n+1
    print(n)

