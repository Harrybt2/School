import random
import sys

# This may come in handy...
from fermat import miller_rabin

# If you use a recursive implementation of `mod_exp` or extended-euclid,
# you recurse once for every bit in the number.
# If your number is more than 1000 bits, you'll exceed python's recursion limit.
# Here we raise the limit so the tests can run without any issue.
# Can you implement `mod_exp` and extended-euclid without recursion?
sys.setrecursionlimit(4000)

# When trying to find a relatively prime e for (p-1) * (q-1)
# use this list of 25 primes
# If none of these work, throw an exception (and let the instructors know!)
primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


# Implement this function
#time complx: O(n^3) Space complx: O(n^2)
def ext_euclid(a: int, b: int) -> tuple[int, int, int]:
    """
    The Extended Euclid algorithm
    Returns x, y , d such that:
    - d = GCD(a, b)
    - ax + by = d

    Note: a must be greater than b
    """
    if b == 0:
        return(1,0,a)
    (x_prime, y_prime, inverse) = ext_euclid(b, a % b) # a recursive operation on n performed n times makes space compexity O(n^2)
    return(y_prime, x_prime - a//b * y_prime, inverse) #division in a recursive run n times makes time complx O(n^3)



# Implement this function
# time complx O(n^2), space complx: O(n^4)
def generate_large_prime(bits=512) -> int:
    """
    Generate a random prime number with the specified bit length.
    Use random.getrandbits(bits) to generate a random number of the
     specified bit length.
    """
    k = 1
    while 1:
        x = random.getrandbits(bits)
        find_prime = miller_rabin(x, k) # miller rabin dominates both complexities
        if find_prime == 'prime':
            return x
       # Guaranteed random prime number obtained through fair dice roll


# Implement this function
# space complx: # O(n^2) time complx: O(n^4)
def generate_key_pairs(bits: int) -> tuple[int, int, int]:
    """
    Generate RSA public and private key pairs.
    Return N, e, d
    - N must be the product of two random prime numbers p and q
    - e and d must be multiplicative inverses mod (p-1)(q-1)
    """
    # find two prime numbers
    p = generate_large_prime(bits)# generate accesses miller rabin which dominates both complexities
    q = generate_large_prime(bits)
    N = p*q # determine what our N is for the key
    
    for i in primes:
        if (p-1)*(q-1) % i != 0:
            e = i
            break
        if i == primes[-1] and (p-1)*(q-1) % i == 0:
            raise Exception('none of the numbers worked for e') 
        
    # now find d
    _, d ,_= ext_euclid((p-1)*(q-1), e)
    d = d % ((p-1)*(q-1)) # this makes sure that d is always a positive val


    return N, e, d
