# Sept 13, 2024 Harrison Denning
# C S 312 Hw 2 prblms 2-3
# Code to perform Modular exponentiation

import numpy as np

def modexp(x, y, N): #input two n-bit integers x and N and the exponent y
    if y == 0:
        return 1
    z = modexp(x, y//2, N)
    if y % 2 == 0:
        return z**2 % N
    else:
        return x * z**2 % N
    
a = int(input('input the base number'))
b = int(input('input the exponent'))
c = int(input('input the mod value'))

result = modexp(a, b, c)
print('result is', result, 'mod(',c,')')