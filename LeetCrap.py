import math
# these are those stupid leet code problems people who want jobs in CS have to know how to do
# eh, maybe it'll be fun (but maybe not)
# I'M AN ENGINEER DANGIT!!

# solve the N-th tribonacci problem
def tribonacci(n): # given a number n, solve the tribonacci sequence
    T = [0, 1, 1]
    for j in range(0, n-2):
        T.append(T[j]+T[j+1]+T[j+2]) 
    return T[-1]

# tribonacci(25)

def two_sums(nums,target): # a target and list are given, find the two numbers in the list that will add up to the target
    how_close = 0
    for n in range(0,len(nums)): # go through every array twice
        potential = nums[n]
        if target >= 0:
            to_find = target - potential
            for j in range(n+1, len(nums)): # go through a second time, don't worry about what you just found
                potential = nums[j]
                if to_find == potential:
                    return [n,j]
        
        else:
            

            to_find = target - potential
            for j in range(n+1, len(nums)): # go through a second time, don't worry about what you just found
                potential = nums[j]
                if to_find == potential:
                    return [n,j]            
# print(two_sums([18,-12,3,0],6))

def combo_recursion(candidates, target, sol, solutions_list, start_index):
    if target == 0:
        solutions_list.append(sol[:])  # Append a copy of sol
        return
    elif target < 0:
        return
    else:
        for n in range(start_index, len(candidates)):
        
            potential = candidates[n]
            # Create new target and sol for recursive call
            new_target = target - potential
            sol.append(potential)
            combo_recursion(candidates, new_target, sol, solutions_list, n)
            sol.pop()  # Backtrack to explore other possibilities

def combination_sum(candidates, target): # this solution is basically a recursion looking at every possiblity on a tree
    solution_list = []
    candidates.sort()
    combo_recursion(candidates, target, [], solution_list, 0)
    return solution_list  # Return the list of solutions

# Example usage
# print(combination_sum([8,7,4,3], 11))

def manhattan_dist(point1, point2):
    return abs(point1[0]-point2[0]) + abs(point1[1] - point2[1])

def min_connecting_cost(points):
    total_cost = 0
    included = set()
    graph = []
    # get a list to create a graph of distances between points
    for i in range(0,len(points)):
        row = []  # Create a new row for the current point
        for j in range(0,len(points)):
            dist = manhattan_dist(points[i], points[j])
            row.append(dist)  # Append the distance to the row
        graph.append(row)  # Append the completed row to the graph
    # until your set contains all points
    # choose a starting point, look across the row at everything it goes to
    # choose the minimum distance and add that to the set
    # go to a new point that is not already in the set and find the min distance to a point
    # repeat
    start_point = 0
    notincluded = set(range(len(points))) # make a set containing everything not included in the tour
    notincluded.remove(start_point) # remove the start from your notincluded
    included.add(start_point) # add it to the included
    while len(included) < len(points): # the set does not contain every point, do prim's algorithm
        # if start_point in included:
        #     start_point = start_point + 1 # increment the counter
        # else:
        # included.add(start_point)
        min_dist = math.inf # create an arbitrary start point
        point_explored = None
        for vertices in notincluded:
            for row in included:
                if graph[vertices][row] <= min_dist and graph[vertices][row]!=0: # find the smallest non-zero distance
                    min_dist = graph[vertices][row]
                    point_explored = vertices
                    # vertice_explored = vertices
        included.add(point_explored)
        notincluded.remove(point_explored)
        # graph[vertice_explored][point_explored] = 0# make sure you don't ever go back to a point in the set
        # graph[point_explored][vertice_explored] = 0
        total_cost = total_cost + min_dist
        start_point = start_point + 1 # increment the counter

        


    return total_cost

# print(min_connecting_cost([[0,0],[1,1],[1,0],[-1,1]]))
def province_recursion(original_start, starting_point, connections, discovered, undiscovered, province_counter):
    if len(discovered) == len(connections):# if all cities have been discovered
        province_counter = province_counter + 1
        # new_city_found = False
        return province_counter
    # or if this city doesnt connect to anything new
    # terminate this search and go back
    for cities in range(0,len(connections[starting_point])): # look at all the potential connections
        new_city_found = False
        if cities in discovered:
            continue
        elif connections[starting_point][cities] == 1: # if the city is connected, explore it
            new_city_found = True
            discovered.add(cities)
            undiscovered.remove(cities)
            province_counter = province_recursion(original_start,cities, connections, discovered, undiscovered, province_counter)
            new_city_found = False
    # if another city wasn't found that connects then add one to your province counter
    if len(discovered) == len(connections):# if all cities have been discovered
        # new_city_found = False
        return province_counter
    elif original_start == starting_point and new_city_found == False:
        # new_city_found = False
        province_counter = province_counter + 1
    return province_counter
def num_provinces(isConnected):
    # run a depth first search
    undiscovered = set(range(len(isConnected))) # keep track of what is yet to be found
    discovered = set() # keep track of everything found til now

    provinces = 0

    while len(discovered) != len(isConnected): #until all cities are accounted for
        orignial_starting_city = next(iter(undiscovered))# start at an arbitrary point
        discovered.add(orignial_starting_city)
        undiscovered.remove(orignial_starting_city)
        provinces = province_recursion(orignial_starting_city, orignial_starting_city, isConnected, discovered, undiscovered, provinces)

    return provinces

# connections = [[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1]]
# print(num_provinces(connections))


def triangle_recursion(triangle, baseline_cost, cost, costs,chosen_index, start_level):
    # for levels in range(start_level,len(triangle)): # start at the first level and work your way down
    for costs in [chosen_index, chosen_index+1]: # go through every cost in it
        new_cost = cost + triangle[start_level][costs]
        if new_cost >= baseline_cost:
            continue # choose the next index
        elif start_level == len(triangle):# if you're at the bottom and found a new solution
            if new_cost < baseline_cost:
                baseline_cost = cost
            else:
                continue
        else:
            triangle_recursion(triangle, baseline_cost, new_cost, costs, chosen_index= costs, start_level=start_level+1)
    return baseline_cost
def min_triangle_path(triangle):
    baseline_cost = triangle[0][0]
    chosen_index = 0
    # if the triangle is one deep just return it
    if len(triangle) == 1:
        return baseline_cost
    # get a first greedy option
    for levels in range(1,len(triangle)): # start at the first level and work your way down
        current_min = math.inf
        for costs in [chosen_index, chosen_index+1]: # go through every cost in it
            if triangle[levels][costs] < current_min:
                chosen_index = costs
                current_min = triangle[levels][costs]
        baseline_cost = baseline_cost + current_min
    
    chosen_index = 0
    cost = triangle[0][0] # start with cost of fisrt item
    # now do a branch and bound
    start_level = 1
    cost = triangle_recursion(triangle, baseline_cost, cost, costs, chosen_index, start_level)
   


    return cost



def better_recursion(triangle, search_row, chosen_index, cost, new_cost):
    if search_row == len(triangle): # if you hit the bottom return
        if cost < new_cost:
            return cost
        else:
            return new_cost
    for costs in [chosen_index, chosen_index+1]: # go through every cost in it
        added_cost = cost + triangle[search_row][costs]
        new_cost = better_recursion(triangle, search_row= search_row+1, chosen_index=costs, cost=added_cost, new_cost= new_cost)
    return new_cost


def better_triangle(triangle):
    baseline_cost = triangle[0][0]
    search_row = 1
    chosen_index = 0
    # if the triangle is one deep just return it
    if len(triangle) == 1:
        return baseline_cost
    # do a depth first search
    new_cost = math.inf
    cost = better_recursion(triangle,search_row, chosen_index, baseline_cost, new_cost)

    return cost


def minimumTotal(triangle):
    if len(triangle) == 1:
        return triangle[0][0]
    
    for row in range(len(triangle)-2, -1, -1):
        for col in range(len(triangle[row])): # go across every element in that row
            cost = triangle[row][col]
            triangle[row][col] = cost  + min(triangle[row+1][col], triangle[row+1][col+1])
    return triangle[0][0]
    # # Start from the bottom
    # for row in range(len(triangle) - 2, -1, -1):
    #     for col in range(len(triangle[row])):
    #         # Update each element to the sum of itself and the minimum of the two elements below
    #         triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1])
    
    # # The top element contains the minimum path sum
    # return triangle[0][0]

# Example
triangle = [[-1], [2, 3], [1, -1, -3]]
print(minimumTotal(triangle))  # Output: -1

# triangle = [[-1],[2,3],[1,-1,-3]]
# print(better_triangle(triangle))