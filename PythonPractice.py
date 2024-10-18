# this is a comment in python
# this is a genetic algorithm to guess a phrase
import string
import random
string.ascii_letters


target = "Target"
guess_length = len(target)


def initial_guess():
    guess = ""
    while len(guess) < guess_length:
        guess = ''.join((guess, random.choice(string.ascii_letters)))
    return guess

def genetic_algorithm():
    first_gen = []
    while len(first_gen) < 10:
        first_gen.append(initial_guess)
    print(first_gen)

def fitness_function(initial_guess):
    i = 0
    fitness = 0
    for letters in target:
        if target[i] == initial_guess[i]:
            fitness += 1
        i += 1
    return fitness


generations = genetic_algorithm()
output = initial_guess()
how_good = fitness_function(initial_guess())
print(generations)
print(output)
print(how_good)