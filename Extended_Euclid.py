# this solves the extended euclid algorithm

def ext_euc(a,b):
    # non-extended euclid
    # if b == 0:
    #     return(a)
    # return ext_euc(b, a%b)

    #extedbded euclid
    if b == 0:
        return(1,0,a)
    (x_prime, y_prime, inverse) = ext_euc(b, a % b)
    return(y_prime, x_prime - a//b * y_prime, inverse)

x = int(input('mod#'))
y = int(input('# to evaluate'))

print(ext_euc(x,y))