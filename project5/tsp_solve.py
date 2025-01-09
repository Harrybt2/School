import math
import random
import copy

from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree
from math import inf
import heapq

def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]
    
def reduce_matrix(matrix): # takes in a matrix, returns the reduced cost matrix
    # look in all the rows, see if there is a zero, if there is, skip it  and remember where the 0 is
    # otherwise subtract the smallest value from all of them, remember that value
    # zeros_in = []
    has_zero = False
    cost_incurred = 0
    
    for row in range(0, len(matrix[1])):
        has_zero = False
        current_min = inf
        inf_count = 0
        for column in range(0,len(matrix)):
            if matrix[row][column] == 0:
                has_zero = True
                # zeros_in.append([row, column])
                break
            elif matrix[row][column] == inf:
                inf_count = inf_count + 1
            #   add in conditions for if a row is all infinity
            else:
                if matrix[row][column] < current_min:
                    current_min = matrix[row][column]
        if has_zero == True:
            continue
        elif inf_count == len(matrix[row]):
            continue
        else:
            for n in range(0, len(matrix[row])):
                matrix[row][n] = matrix[row][n] - current_min
            cost_incurred = cost_incurred + current_min

            
    # any column that doesnt have a zero, subrtract the smallest number from it
    # remember that value subtracted
    """Add condition for all being infinity"""
    for column in range(0, len(matrix)):
        current_min = inf
        has_zero = False
        inf_count = 0
        for row in range(0, len(matrix[1])):
            if matrix[row][column] == 0:
                has_zero = True
                # zeros_in.append([row, column])
                break
            elif matrix[row][column] == inf:
                inf_count = inf_count + 1
            else:
                if matrix[row][column] < current_min:
                    current_min = matrix[row][column]
        if has_zero == True:
            continue
        elif inf_count == len(matrix):
            continue
        else:
            for n in range(0, len(matrix)): # until you've gone down the whole row
                matrix[n][column] = matrix[n][column] - current_min
            cost_incurred = cost_incurred + current_min
    return matrix, cost_incurred

def greedy_best(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    # put inf on the diagonals to begin so my code works
    for j in range(0,len(edges)):
        edges[j][j] = inf
    stats = []
    
    start_point = 0
    
    best_tour = []
    best_total = math.inf
    
    for starts in range(0,len(edges)):
        start = starts
        tour = [start]
        total_cost = 0
        for n in range(0,len(edges)): # you'll do it as many times as you have cities
            min_cost = math.inf
            # here are some exit conditions
            if len(tour) == len(edges): # if you've visited everywhere
                if edges[tour[-1]][starts] < inf: # if you're last point can connect to your first point
                    total_cost += edges[tour[-1]][starts] # add in its cost and return it as a solution
                    if total_cost < best_total: # if its a better greedy solution
                        best_total = total_cost
                        best_tour = tour
                    # stats.append(SolutionStats(
                    # tour=tour,
                    # score=total_cost,
                    # time=timer.time(),
                    # max_queue_size=1,
                    # n_nodes_expanded=0,
                    # n_nodes_pruned=0,
                    # n_leaves_covered=0,
                    # fraction_leaves_covered=0))

                    # return stats # if you found a valid solution then get out of this!
                    break
                else: 
                    break
                    # otherwise you need to try a new starting point


            for city, cost in enumerate(edges[start]):
                if cost < min_cost and city not in tour: # if its cheaper to go there and you've not been there before
                    min_cost = cost
                    start = city
            if start in tour: # if you got stuck
                break
            total_cost += min_cost
            tour.append(start)
        
    stats.append(SolutionStats(
                    tour=best_tour,
                    score=best_total,
                    time=timer.time(),
                    max_queue_size=1,
                    n_nodes_expanded=0,
                    n_nodes_pruned=0,
                    n_leaves_covered=0,
                    fraction_leaves_covered=0))

    return stats # if you found a valid solution then get out of this!



def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    # put inf on the diagonals to begin so my code works
    for j in range(0,len(edges)):
        edges[j][j] = inf
    stats = []
    
    start_point = 0
    
    best_tour = []
    best_total = math.inf
    
    for starts in range(0,len(edges)):
        start = starts
        tour = [start]
        total_cost = 0
        for n in range(0,len(edges)): # you'll do it as many times as you have cities
            min_cost = math.inf
            # here are some exit conditions
            if len(tour) == len(edges): # if you've visited everywhere
                if edges[tour[-1]][starts] < inf: # if you're last point can connect to your first point
                    total_cost += edges[tour[-1]][starts] # add in its cost and return it as a solution
                    
                    stats.append(SolutionStats(
                    tour=tour,
                    score=total_cost,
                    time=timer.time(),
                    max_queue_size=1,
                    n_nodes_expanded=0,
                    n_nodes_pruned=0,
                    n_leaves_covered=0,
                    fraction_leaves_covered=0))

                    return stats # if you found a valid solution then get out of this!
                    
                else: 
                    break
                    # otherwise you need to try a new starting point


            for city, cost in enumerate(edges[start]):
                if cost < min_cost and city not in tour: # if its cheaper to go there and you've not been there before
                    min_cost = cost
                    start = city
            if start in tour: # if you got stuck
                break
            total_cost += min_cost
            tour.append(start)
        
    stats.append(SolutionStats(
                    tour=best_tour,
                    score=best_total,
                    time=timer.time(),
                    max_queue_size=1,
                    n_nodes_expanded=0,
                    n_nodes_pruned=0,
                    n_leaves_covered=0,
                    fraction_leaves_covered=0))

    return stats # if you found a valid solution then get out of this!


def better_recursion(edges, discovered, cost, best_cost, search_row, best_tour,start_row,stats,timer):
    # print(timer.time_out())
    if timer.time_out() == True:# once you time out then stop
        return best_cost, best_tour
    if len(discovered) == len(edges): # if you hit the bottom return
        if edges[discovered[-1]][start_row] < math.inf: # if the last node can connect to the start
            cost += edges[discovered[-1]][start_row] # then add that cost 
            if cost < best_cost:

                return cost, discovered[:]
            else:
                return best_cost, best_tour
        else:
            return best_cost, best_tour
    for city in range(0,len(edges[search_row])): # go through every connection to a city
        if city in discovered: # if you've been to the city on your tour, ignore it
            continue
        else:
            if edges[search_row][city] < math.inf:
                discovered.append(city) # add that city to your tour
                cost = cost + edges[search_row][city] # add its cost to your running total
                best_cost , best_tour= better_recursion(edges, discovered, cost, best_cost, city, best_tour, start_row, stats,timer)
                discovered.pop()
                cost -= edges[search_row][city]
    return best_cost, best_tour

def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    # choose a starting point, explore all the connections 
    stats = []
    discovered = []
    best_tour = []
    start_point = 0
    discovered.append(start_point)
    cost = 0
    best_cost, best_tour = better_recursion(edges, discovered, 0,math.inf, start_point, best_tour, start_point, stats,timer)
    
    stats.append(SolutionStats(
                tour=best_tour,
                score=best_cost,
                time=timer.time(),
                max_queue_size=1,
                n_nodes_expanded=0,
                n_nodes_pruned=0,
                n_leaves_covered=0,
                fraction_leaves_covered=0
            ))
    # use the greedy dfs as first iteration, keeping track of which branches you've taken
    # explore every possible path (or else you time out)
    return stats

graph = [
    [0, 7, 3, 12],
    [3, 0, 6, 14],
    [5, 8, 0, 6],
    [9, 3, 5, 0],
    ]
print(dfs(graph, timer=Timer()))

def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    """
    Search(PQ):
        S = {PQ}
        BSSF = None
        while S is not empty:
            p = S.next()
            if is_solution(P) and P < BSSF:
                BSSF = P
            for child in expand(P):
                if lower_bound(child) < BSSF:
                    S.add(child)
                return BSSF

    """
    

    stats = []
    
    greedy_stat = greedy_tour(edges, timer=Timer())
    bssf = (greedy_stat[0].tour,greedy_stat[0].score)
    print(bssf)
    reduced_matrix, cost_to_reduce = reduce_matrix(copy.deepcopy(edges))
    stack = [([0], cost_to_reduce)] # stack is a list where every value is a touple of a list and its cost
    while len(stack)>0:
        P = stack.pop()
        # print(timer.time_out())
        if timer.time_out() == True:# once you time out then stop
            stats.append(SolutionStats(
            tour=bssf[0],
            score=bssf[1],
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=0,
            n_nodes_pruned=0,
            n_leaves_covered=0,
            fraction_leaves_covered=0
                ))        
            return stats
        if len(P[0]) == len(edges): # if everything is in your tour
            if P[1] < bssf[1]: # if the cost is less than your current best
                bssf = P # then store the tour and its value
        else:
            for j, connections in enumerate(reduced_matrix[P[0][-1]]): # look through what the last thing in you current tour connexts to
                if j not in P[0] and connections < math.inf: # if it actually connects and isn't a repeat
                    child = P[0]+[j]# then make it a child
                    cost = P[1] + connections # add its cost
                    if cost <= bssf[1]: # if its not worse, add it to the stack
                        stack.append((child, cost))
                    # else:
                    #     cost-= connections
    stats.append(SolutionStats(
            tour=bssf[0],
            score=bssf[1],
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=0,
            n_nodes_pruned=0,
            n_leaves_covered=0,
            fraction_leaves_covered=0
        ))        
    return stats


def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []

    greedy_stat = greedy_best(edges, timer=Timer())
    bssf = (0,greedy_stat[0].tour,greedy_stat[0].score)
    print(bssf)
    reduced_matrix, cost_to_reduce = reduce_matrix(copy.deepcopy(edges))
    stack = []
    # stack = [([0], cost_to_reduce, reduced_matrix)]
    heapq.heappush(stack, (0, [0],cost_to_reduce, reduced_matrix))  # (cost, tour, reduced_matrix)
     # stack is a list where every value is a touple of a list and its cost, and its reduced cost matrix
    while stack: # until the priority queue is empty
        # print('popping off next')
        # for possiblities in stack:
        P = heapq.heappop(stack)
        # print(timer.time_out())
        if timer.time_out() == True:# once you time out then stop
            stats.append(SolutionStats(
            tour=bssf[1],
            score=bssf[2],
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=0,
            n_nodes_pruned=0,
            n_leaves_covered=0,
            fraction_leaves_covered=0
                ))        
            return stats
        if len(P[1]) == len(edges): # if everything is in your tour
            if P[2] < bssf[2]: # if the cost is less than your current best
                bssf = P # then store the tour and its value
        else:
            for j, connections in enumerate(reduced_matrix[P[1][-1]]): # look through what the last thing in you current tour connexts to
                if j not in P[1] and connections < math.inf: # if it actually connects and isn't a repeat
                    # print('child being expanded')
                    # then alter the matrix and reduce it again
                    child = P[1]+[j]# then make it a child
                    child_reduced = copy.deepcopy(P[3])
                
                    for items in range(0,len(edges)):
                        child_reduced[child[-2]][items] = math.inf# make the row of where you started inf
            
                    for items in range(0,len(edges)):
                        child_reduced[items][child[-1]] = math.inf
                   
                    child_reduced[child[-1]][child[-2]] = math.inf


                    child_reduced, cost_to_reduce = reduce_matrix(child_reduced) # down each of the columns for the children you have
                    # store the reduced cost matrix in P
                    # make the column of where you went to inf
                    
                    cost = P[2] + connections + cost_to_reduce # add its cost
                    if cost <= bssf[2]: # if its not worse, add it to the stack
                        # stack.append((child, cost, child_reduced))
                        heapq.heappush(stack, (cost_to_reduce-len(child), child, cost, child_reduced)) 
                        """Add it in the heap based on cost to reduce minus the length of the tour"""
                        # print('changed bssf')
                    # else:
                    #     cost-= connections
    stats.append(SolutionStats(
            tour=bssf[1],
            score=bssf[2],
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=0,
            n_nodes_pruned=0,
            n_leaves_covered=0,
            fraction_leaves_covered=0
        ))        
    return stats
graph = [
    [0, 7, 3, 12],
    [3, 0, 6, 14],
    [5, 8, 0, 6],
    [9, 3, 5, 0],
    ] # expect 15
# graph = [
#     [0, 10, 15, 20],  # Distances from city 0 to other cities
#     [10, 0, 35, 25],  # Distances from city 1 to other cities
#     [15, 35, 0, 30],  # Distances from city 2 to other cities
#     [20, 25, 30, 0],  # Distances from city 3 to other cities
# ] # expect 80
# graph = [
#     [0, 29, 20, 21, 16, 31],  # Distances from city 0
#     [29, 0, 15, 29, 28, 40],  # Distances from city 1
#     [20, 15, 0, 15, 14, 25],  # Distances from city 2
#     [21, 29, 15, 0, 25, 30],  # Distances from city 3
#     [16, 28, 14, 25, 0, 18],  # Distances from city 4
#     [31, 40, 25, 30, 18, 0],  # Distances from city 5
# ]
# graph = [
#     [0, 10, 15, 20, 25],  # Distances from city 0
#     [12, 0, 35, 25, 17],  # Distances from city 1
#     [14, 19, 0, 30, 22],  # Distances from city 2
#     [18, 24, 33, 0, 16],  # Distances from city 3
#     [21, 28, 23, 14, 0],  # Distances from city 4
# ]


print(branch_and_bound_smart(graph, timer=Timer()))

   