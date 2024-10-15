"""A Genetic Algorithm to Find Optimal Wind Turbine Blade Shapes"""
# Harrison Denning October 8, 2024
"""
General Idea:
Make a Genetic algorithm which finds the highest lift to drag coeff. over the windspeed we want.
Give the GA all the airfoils you want to consider and find the best combination.
This is designed to be used with 'pyhton-turbine-blade-designer' by Nicola Sorace to retrun C_l/C_d ratio (Coefficients of 
lift to drag). The Polars and other data on airfoils is gotten through running Xfoil through python, or else you can just 
manually download all the airfoil shapes off something like airfoiltools.com


PsuedoCode:
Here I will go through each function in sudo code and describe what its doing

#### This function receives a list of airfoil names, used to create a bunch of children, each child a combination of a few
airfoils. These combinations are all run through the blade performance function and their fitnesses are returned. A new
generation of possibilities is produced and these are evaluated for their fitness, repeat this until your convergence
condition is met, either the best doesn't change over a few iterations or you just run a whole bunch then stop.###


Inputs: List of airfoil names
Output: The airfoil shape that gives the best C_l/ C_d ratio
def genetic_Alg(Inputs):
    init_pop = generate_founders(Inputs)
    eval_fit = blade_perf(init_pop)
    While not converged:
        new_gen = generate_gen(eval_fit)
        eval_fit = blade_perf(new_gen)
    return best airfoils

### This is the blade performance evaluator function and our fitness function. you pass in each of the children created before
and return their fitness. This works using the 'pyhton-turbine-blade-designer' code by Nicola Sorace found in github. This
program outputs a bunch of plots but what we hope to get out of it is the average Lift to Drag ratio over the windspeeds we're
interested in. We'll take the negative of this (convention in optimization is to alwys minimize) and return it. The blade
performance function calls Sorace's code and gets all the C_l and C_d's over the interested wind speeds, the fitness function
calculates the value to minimize (the ratio of the average) 

A potential tricky part is to get the airfoil shapes to input or configure a .yaml file that Sorace's code reads###

Input: One airfoil combination
Output: fitness value

def blade_perf(airfoil_shape):
    yaml_file = create/convert_yaml(airfoil_shape)
    C_l, C_d = run_Sorace_code(yaml_file)
    fit = fitness(C_l,C_d)
    return fit

def fitness(Lift, Drag)
    fitness_val = rms(Lift) / Rms(Drag) # Take the root mean square of the values, you shouldn't get negatives, but just in case
    return fitness_val

###The rest of these functions are the nitty-gritty of the Genetic Algorithm. It's not too difficult, but if you've never dealt
with them before it can get confusing and I would recomend watching a youtube video or two. ###

### first is the function to generate that initial population. You should input all of the airfoil numbers that you want the code
to consider. Then you just go through and randomly combine them into lists of 2 or more airfoils (make sure all the code,
especially the Sorace stuff is in agreement about how many airfoil shapes to expect). Also decide how many different combinations
you want to create.

input list of airfoil names, # of parents wanted
output a list of lists of random airfoils
def generate_founders(Airfoil_names, n):
    i = 0
    parents = []
    while i < n: # repeat this until you generate as many parents as you want
        j = 0
        airfoils = []
        while j < 3: # if you change the number of foils then change what the yaml creater expects
            airfoils.append(Airfoil_names[randomint]) # this should input a random foil into the list each time
            j = j + 1
        parents.append(airfoils) # this puts the new parent into the parents list
        i = i + 1
    return parents

### this  function generates the children. Once the founder's (or previous generation's) fitnesses have all been evaluated
we use this information to create a bunch of children. There are MANY ways you can decide all this in genetic algorithms. First
is a parent select to decide which from the previous generation will be recombined to make new children. We'll just do a
tournament style selection. This is good because it insures the best one always passes to the next generation. We select two
random ones from our list and pass the one with the lowest fitness value. Then we breed all the selected parents to make new
children. Children will be a combination of the parents (obviously). We'll split each parent at a random index and splice them
together to make a child. There should also be some mutation present to avoid local minima. This randomly changes an airfoil
at a certain probability (it should do this swap not often). Once these children are all made it outputs another list of lists
of airfoil shapes to try again!###

inputs list of parents and their fitness vals
output list of children
def generate_children(parents, fit_vals, airfoils_list):
    children = []
    # pass te best one immediately to the children
    best_one = fit_vals[0]
    for i in range (1,len(fit_vals)): #iterate over array
        if fit_vals[i] > best_one: #to check max value
            best_one = list1[i]
            ind = i
    children.append(parents[ind])

    # find the breeders that we'll use to make more children
    breeders = tournament_select(parents, fit_vals)
    while len(children) < 50; # change to make as many kids as you want
        parent1 = breeders[randint] # make sure the int is in the range of index
        parent2 = breeders[randint]

        splice_at = randint # make sure the int is in the range of index
        child = parents1[:splice_at] + parent2[splice_at:]
        
        # mutate on a probaility
        prob = 0.1
        num = random(0,1) # make a random number btween 0 and 1
        if num < prob:
            child[randint] = airfoils_list[randint] # swap it out with a totally new airfoil
            
        children.append(child)


    return children

def tournament_select(contenders, worthiness):
    champions = [] # here we will store the winners (breeders)
    while len(contenders) != 0: # until you've gone through them all
        if len(contenders) = 1:
            champions.append(contenders.pop(0)) # if there's one left then add it to the breeders and remove it
            break
        first_contender = randomint # make sure its in the range of the index
        second_contnder = randomint
        if worthiness(first_contender) < worthiness(second_contender): # remember smallest wins
            champions.append(contenders.pop(first_contender)) # remove it from contenders and into champions
        else:
            champions.append(contenders.pop(second_contender))
    return champions


        

"""
import numpy as np
from numpy.random import rand, randint

def genetic_Alg(airfoil_names):

    # generate an initial population, decide how many in pop
    init_pop = generate_founders(airfoil_names, 5)
    eval_fit = []
    for blade in init_pop:
        eval_fit.append(blade_perf(blade)) # check their fitness

    # this is all to get the highest fitness airfoil combination to compare in subsequent gen. for convergence
    index_of_best = eval_fit.index(min(eval_fit))
    old_best = init_pop[index_of_best]
    old_gen = init_pop

    n = 0 # start the counter
    while n < 5 : #converge if the best value hasn't changed after 5 iterations
        # create a new generation and find their fitnesses
        new_gen = generate_children(old_gen, eval_fit, airfoil_names)
        eval_fit = []
        for blade in new_gen:
            eval_fit.append(blade_perf(blade)) # check their fitness

        # look at the best airfoil combination
        index_of_best = eval_fit.index(min(eval_fit))
        new_best = new_gen[index_of_best]
        old_gen = new_gen
        if old_best == new_best: # if the best one is the same as last time incremement the counter
            n = n + 1
        else: # if its not then reset it, we'll keep going until the best is the same 5 times in a row
            n = 0
        old_best = new_best
    # now that we've concerged take the top 3 airfoils
    top_3_indices = np.argsort(eval_fit)[0:3]
    top_3_airfoils = [new_gen[i] for i in top_3_indices]

    return top_3_airfoils

## there's no sudo code for this, ChatGPT wrote it, it just edits the YAML so we can run Sorace's code and get the C_l and C_d
def modify_yaml(airfoils):
    file_path = 'config.yaml'
    # Ensure there are exactly three strings
    if len(airfoils) != 3:
        raise ValueError("You must provide exactly three strings.")

    with open(file_path, 'r') as file:
        # Read the YAML file lines into a list
        lines = file.readlines()

    # Check if the file has at least 32 lines
    if len(lines) < 32:
        raise ValueError("The YAML file has fewer than 32 lines.")

    # Replace lines 23, 29, and 32 with the new strings (lines are 0-indexed)
    lines[22] = '    ' + airfoils[0] + '\n'  # Line 23
    lines[28] = '    ' + airfoils[1] + '\n'  # Line 29
    lines[31] = '    ' + airfoils[2] + '\n'  # Line 32

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)
    return file_path


def blade_perf(airfoil_shape):
    yaml_file = modify_yaml(airfoil_shape)
    C_l, C_d = run_Sorace_code(yaml_file)
#     C_l = [98.4, 100.2, 101.5, 99.3, 102.8, 97.1, 100.7, 99.8, 101.2, 100.4, 
# 99.9, 98.6, 101.1, 100.6, 102.4, 99.7, 100.1, 101.3, 98.9, 97.5,
# 101.6, 100.3, 99.2, 100.9, 101.7, 98.8, 102.1, 99.5, 100.8, 101.0,
# 98.7, 97.4, 101.8, 100.0, 99.6, 102.3, 98.3, 100.5, 101.4, 100.2,
# 99.0, 101.9, 98.5, 97.3, 102.0, 99.4, 100.4, 101.0, 99.1, 98.2]
# # this mimics the list we hope to get from SOrace's code
#     C_d = [48.1, 51.2, 50.6, 49.8, 50.0, 48.9, 51.1, 49.2, 50.7, 48.5, 
# 49.3, 50.2, 51.5, 49.7, 50.9, 48.6, 49.9, 50.3, 48.7, 50.1,
# 49.0, 51.0, 50.5, 48.4, 49.4, 50.8, 51.3, 48.8, 50.4, 49.1,
# 48.2, 51.4, 50.6, 48.3, 49.5, 50.0, 48.9, 50.7, 49.6, 48.5,
# 51.1, 50.9, 48.6, 49.8, 50.2, 48.7, 51.3, 49.2, 50.4, 49.0]
    fit = fitness(C_l,C_d)
    return fit


"""
## this is a dummy function to debug, delete and replace later ##
def blade_perf(airfoil_shape):
    fitness = randint(-100, 0)
    return fitness
"""

def fitness(Lift, Drag):
    # # Square the differences element-wise
    # squared_diff = [(x - y)**2 for x, y in zip(Lift, Drag)]
    
    # # Calculate the mean of the squared differences
    # mean_squared_diff = np.mean(squared_diff)
    
    # # Take the square root of the mean
    # rms_value =  -np.sqrt(mean_squared_diff) # take the negative because you minimize by convention
    # We want average lift to drag ratio
    mean_lift = np.mean(Lift)
    mean_drag = np.mean(Drag)

    lift_to_drag = -mean_lift/mean_drag
    return lift_to_drag

def generate_founders(Airfoil_names, n):
    i = 0
    parents = []
    while i < n: # repeat this until you generate as many parents as you want
        j = 0
        airfoils = []
        while j < 3: # if you change the number of foils then change what the yaml creater expects
            airfoils.append(Airfoil_names[randint(0,len(Airfoil_names))]) # this should input a random foil into the list each time
            j = j + 1
        parents.append(airfoils) # this puts the new parent into the parents list
        i = i + 1
    return parents

# airfoils = ['one', 'two', 'three' , 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
# print(generate_founders(airfoils, 10))


def generate_children(parents, fit_vals, airfoils_list):
    children = []
    # pass te best one immediately to the children
    index_of_best = fit_vals.index(min(fit_vals))
    children.append(parents[index_of_best])

    # find the breeders that we'll use to make more children
    breeders = tournament_select(parents, fit_vals)
    while len(children) < 5: # change to make as many kids as you want
        parent1 = breeders[randint(0, len(breeders))] # make sure the int is in the range of index
        parent2 = breeders[randint(0, len(breeders))]

        splice_at = randint(0, len(parent1)) # make sure the int is in the range of index
        child = parent1[:splice_at] + parent2[splice_at:]
        
        # mutate on a probaility
        prob = 0.1
        num = rand() # make a random number between 0 and 1
        if num < prob:
            child[randint(0, len(child))] = airfoils_list[randint(0, len(airfoils_list))] # swap it out with a totally new airfoil
            
        children.append(child)


    return children

def tournament_select(contenders, worthiness):
    champions = [] # here we will store the winners (breeders)
    while len(contenders) != 0: # until you've gone through them all
        if len(contenders) == 1:
            champions.append(contenders.pop(0)) # if there's one left then add it to the breeders and remove it
            break
        if len(contenders) == 2: # we need a special case here
            first_contender_norm = 0
            second_contender_norm = 1
        else:
            first_contender = randint(0, len(contenders)-1) # make sure its in the range of the index
            second_contender = first_contender - randint(1, len(contenders)-1) # this insures you never chose the same contender for both
            # negative indices will screw us up later so fix it now
            first_contender_norm = first_contender if first_contender >= 0 else len(contenders) + first_contender
            second_contender_norm = second_contender if second_contender >= 0 else len(contenders) + second_contender

        if worthiness[first_contender_norm] < worthiness[second_contender_norm]: # remember smallest wins
            champions.append(contenders[first_contender_norm]) # remove it from contenders and into champions
            # remove it from the worthiness list so it still maps
            # Check which index is larger and delete it first
            if first_contender_norm > second_contender_norm:
                del contenders[first_contender_norm]
                del worthiness[first_contender_norm]
                del contenders[second_contender_norm]
                del worthiness[second_contender_norm]
            else:
                del contenders[second_contender_norm]
                del worthiness[second_contender_norm]
                del contenders[first_contender_norm]
                del worthiness[first_contender_norm]
        else:
            champions.append(contenders[second_contender_norm])
            # Check which index is larger and delete it first
            if first_contender_norm > second_contender_norm:
                del contenders[first_contender_norm]
                del worthiness[first_contender_norm]
                del contenders[second_contender_norm]
                del worthiness[second_contender_norm]
            else:
                del contenders[second_contender_norm]
                del worthiness[second_contender_norm]
                del contenders[first_contender_norm]
                del worthiness[first_contender_norm]

    return champions

airfoils = ['one', 'two', 'three' , 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
# values = [-10,-12,-5,-8,-6,-14,-7,-8,-2,-15]
print(genetic_Alg(airfoils))