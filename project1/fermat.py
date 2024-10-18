import argparse
import random


# This is a convenience function for main(). You don't need to touch it.
def prime_test(N: int, k: int) -> tuple[str, str]:
    return fermat(N, k), miller_rabin(N, k)


# You will need to implement this function and change the return value.
# time complexity O(n^3),space complexity O(n^2)
def mod_exp(x: int, y: int, N: int) -> int: # the input has size N
    if y == 0:
        return 1
    z = mod_exp(x, y//2, N) # division has complexity O(n^2) run n times needing more space each time
    if y % 2 == 0:
        return z**2 % N
    else:
        return x * z**2 % N


# You will need to implement this function and change the return value.
def fprobability(k: int) -> float:
    return 1/2**k


# You will need to implement this function and change the return value.
def mprobability(k: int) -> float:
    return 1/4**k


# You will need to implement this function and change the return value, which should be
# either 'prime' or 'composite'.
#
# To generate random values for a, you will most likely want to use
# random.randint(low, hi) which gives a random integer between low and
# hi, inclusive.
## Space complexity: O(n^2) time complexty (n^3)
def fermat(N: int, k: int) -> str: # N is an integer being tested for primality, k is a number of rounds to test it over
    g = 0 # complexity is constant
    a = []
    while g < k:
        # don't include 1 or N because a prime is divisible by itslef and 1
        a.append(random.randint(2,N-1)) #complexity here is O(n) because you're adding elements to a list
        g = g + 1

    for x in a:
        if mod_exp(x,N-1,N) != 1: #this recursion in Mod_exp has space complx O(n^2) which dominates
            # the time complexity is O(n^3) from mod_exp as well, it only recurs a constant # times
            return 'composite'
        else:
            break
    return 'prime'
# print(fermat(7741,100))
# You will need to implement this function and change the return value, which should be
# either 'prime' or 'composite'.
#
# To generate random values for a, you will most likely want to use
# random.randint(low, hi) which gives a random integer between low and
# hi, inclusive.
#space complx: O(n^2) time: O(n^4)
def miller_rabin(N: int, k: int) -> str:
    ## Project description Sudo coded:
    if N<3:
        return 'prime'
    if N % 2 == 0:
        return 'composite'
    for _ in range(k):
        x = random.randint(2,N-1)
        n = N-1
        val = mod_exp(x, n, N) # mod_exp run n times brings the complexity to O(n^2) 
        #new var are not created with each loop so the space complexity is not increased and stays at O(n^2)
        while 1:
            if val != 1:
                if val == N-1:
                    break
                return 'composite'
            if n%2 !=0:
                break
            n = n//2
            val = mod_exp(x, n, N)
    return 'prime'



    ## wikepedia Sudo-Coded:
        # n = N-1
        # s = 0
    # while n % 2 == 0:
    #     n = n/2
    #     s = s + 1
    
    # d = n
    # print(2**s*d)
    # g = 0
    
    # a = []
    # while g < k:
    #     a = random.randint(2,N-1) # don't include 1 or N because a prime is divisible by itslef and 1
    #     x = a**d % N
    #     g = g + 1
    #     j = 0
    #     while s < j:
    #         y = x**2 % N
    #         if y == 1 and x != 1 and x != N-1:
    #             return 'composite'
    #     if y != 1:
    #         return 'composite'
    # return 'prime'
# prime_args = [17, 7520681183, 7263570389, 8993337217, 1320230501, 4955627707, 1095542699, 4505853973, 3176051033,
#               6620550763, 2175869827, 565873182758780452445419697353, 529711114181889655730813410547,
#               600873118804270914899076141007, 414831830449457057686418708951, 307982960434844707438032183853]
# for x in prime_args:
#     print(miller_rabin(x, 100))

def main(number: int, k: int):
    fermat_call, miller_rabin_call = prime_test(number, k)
    fermat_prob = fprobability(k)
    mr_prob = mprobability(k)

    print(f'Is {number} prime?')
    print(f'Fermat: {fermat_call} (prob={fermat_prob})')
    print(f'Miller-Rabin: {miller_rabin_call} (prob={mr_prob})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int)
    parser.add_argument('k', type=int)
    args = parser.parse_args()
    main(args.number, args.k)
