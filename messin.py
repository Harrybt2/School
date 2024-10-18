# this is for messing around, implementing small bits of code
import random 
# HW3 prblm 4, search a list to see if there exist A[i] == i

A =[-12,-11,-5,-1,0,6,11,13,24,114]
# for _ in range(1:100):
#     A.append(random.randint())
# print(len(A))
def find_i(list):
    test_index = len(list)//2
    test_val = list[test_index]
    print(len(list))
    if test_index != test_val and len(list) == 1:
        return 'there isn\'t!'
    if test_index == test_val:
        return 'there is!'
    if test_index <test_val:
        return find_i(list[test_index:])
    if test_index > test_val:
        return find_i(list[:test_index])

print(find_i(A))