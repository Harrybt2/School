# Uncomment this line to import some functions that can help
# you debug your algorithm
from plotting import draw_line, draw_hull, circle_point, plot_points
import matplotlib.pyplot as plt
import numpy as np
from generate import generate_random_points
from test_utils import is_convex_hull
import time
## maybe there's a way to do this with classes, but we can just hardcode in base cases for 1,2, and 3 points
# class Node:
#     def __init__(self, x, y):
#         # Initialize the position of the node
#         self.x = x  # X-coordinate
#         self.y = y  # Y-coordinate
        
#         # Upon initialization, the node's clockwise and counterclockwise pointers refer to itself
#         self.cw_node = self  # Clockwise node
#         self.ccw_node = self  # Counterclockwise node

def get_slope(left_point, right_point):
   #print(left_point,right_point)
    slope = (right_point[1] - left_point[1])/(right_point[0]-left_point[0])
    return slope

def compute_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    
    # plt.hold(True)
    points.sort(key=lambda i: i[0]) # sort the points by x value
    plot_points(points)
    my_hull = recursion(points) # run the convex hull on the sorted lisst of points
    plt.show()
    """Return the subset of provided points that define the convex hull"""
    # input('Press Enter')
    return my_hull
## to go down to a one base case store every point with itself as its clockwise neghbor and counter-clockwise neighbor
### Psuedocode for dividing the problem and solving recursively

def recursion(points):
    if len(points) ==1:
        # add code here that combines the LH and RH points in this case
        return points
    if len(points) == 2:
        # just return those points, they are ordered
        return points
    if len(points) == 3: # sort the list to go clockwise, [left most point, bigger slope, smaller slope]
        slope1 = get_slope(points[0],points[1])
        slope2 = get_slope(points[0],points[2])
        if slope1 > slope2:
            return points # if that's the case then its already in CW order
        else:
            points[2], points[1] = points[1], points[2]
            return points

        
    index_to_split_at = len(points)//2
    left_hand_group = points[:index_to_split_at]
    right_hand_group = points[index_to_split_at:] # split them into a left hand group and a right hand group
    # each of these must be split as well until it's down to 3 points in each group
    left_hand_group = recursion(left_hand_group)
    right_hand_group = recursion(right_hand_group)
    # take two neighboring groups and throw them into the common tangents code
    return create_hull(left_hand_group, right_hand_group) # recombine these two with just the edges and look at next one  """

"""" Psuedo code for finding the common tangents"""

##Upper Tangent
def find_upper_tang(LH_group, RH_group):
    furthest_right = LH_group.index(max(LH_group, key=lambda x: x[0])) # """this has to go through it all again and makes a complexity of n, but this is overshadowed by the other complexities"""
    POI_right = RH_group[0]#eft-most point Right-hand group
    POI_left = LH_group[furthest_right]#right-most point in Left-hand Group
    old_slope = get_slope(POI_left, POI_right) # always do get slope with LEFT POINT, RIGHT POINT
    start_pos_L = furthest_right-1
    start_pos_R = 1
        # you need to go through the following until the coordinates of the upper tangent are found
    n = 0
    while n < 5: #run this a finite # of times , 3 is probably enough or else see if the the ppoint hasn't changed after a loop then you're done
        old_point = POI_right # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
        index = (start_pos_R) % len(RH_group)  # Start at the position you specified (clockwise)
        while True:
            points = RH_group[index]
            new_slope = get_slope(POI_left, points)

            if new_slope < old_slope:  # Break condition
                POI_right = old_point
                start_pos_R = RH_group.index(points)
                break  # Exit the loop once the condition is satisfied
            
            old_slope = new_slope
            old_point = points

            # Update index to cycle through LH_group forwards in a circular manner
            index = (index + 1) % len(RH_group)  # This keeps the index in range

        old_point = POI_left # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
        index = start_pos_L   # Start at the position you specified (counter-clockwise)
        while True:
            points = LH_group[index]
            new_slope = get_slope(points, POI_right)

            if new_slope >= old_slope:  # Break condition
                POI_left = old_point
                start_pos_L = LH_group.index(points)
                break  # Exit the loop once the condition is satisfied
            
            old_slope = new_slope
            old_point = points

            # Update index to cycle through LH_group backwards in a circular manner
            index = (index - 1) % len(LH_group)  # This keeps the index in range

        n = n+1

    return  POI_left, POI_right

def find_lower_tang(LH_group, RH_group):

    furthest_right = LH_group.index(max(LH_group, key=lambda x: x[0])) # """this has to go through it all again and makes a complexity of n, but this is overshadowed by the other complexities"""
    POI_right = RH_group[0]#eft-most point Right-hand group
    POI_left = LH_group[furthest_right]#right-most point in Left-hand Group
    old_slope = get_slope(POI_left, POI_right) # always do get slope with LEFT POINT, RIGHT POINT
    

    start_pos_L = furthest_right+1
    start_pos_R = -1
    # you need to go through the following until the coordinates of the upper tangent are found
    n = 0
    while n < 5: #run this a finite # of times , 3 is probably enough or else see if the the ppoint hasn't changed after a loop then you're done
        old_point = POI_right # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
        index = start_pos_R  % len(RH_group) # Start at the position you specified (counter-clockwise)
        while True:
            points = RH_group[index]
            new_slope = get_slope(POI_left, points)

            if new_slope >= old_slope:  # Break condition
                POI_right = old_point
                start_pos_R = RH_group.index(points)
                break  # Exit the loop once the condition is satisfied
            
            old_slope = new_slope
            old_point = points

            # Update index to cycle through LH_group forwards in a circular manner
            index = (index - 1) % len(RH_group)  # This keeps the index in range       

        old_point = POI_left # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
        index = (start_pos_L) % len(LH_group) # Start at the position you specified (clockwise)
        while True:
            points = LH_group[index]
            new_slope = get_slope(points, POI_right)

            if new_slope <= old_slope:  # Break condition
                POI_left = old_point
                start_pos_L = LH_group.index(points)
                break  # Exit the loop once the condition is satisfied
            
            old_slope = new_slope
            old_point = points

            # Update index to cycle through LH_group forwards in a circular manner
            index = (index + 1) % len(LH_group)  # This keeps the index in range

        # for points in LH_group[start_pos_L+1:] + LH_group[:start_pos_L]: # make ure the list is ordered to go clockwise
        #     new_slope = get_slope(points, POI_right)
        #     if new_slope <= old_slope: # if the slope gets les negative doesnt change then we wanted that last point
        #         POI_left = old_point #once you take a new point you don't care about the slope to the last point
        #         start_pos_L = LH_group.index(points)
        #         break # get out of this for-loop because we have the next value in the left-hand group
        #     old_slope = new_slope
        #     old_point = points
        n = n+1

    return POI_left, POI_right
    
## this function receieves two hulls and combines them into one 
def create_hull(LH_group, RH_group):
    if len(LH_group) == 1:
        hull_points = LH_group + RH_group # in this case the lefft most can just be added in and it will maintain its clockwiseness
        return hull_points
    if len(RH_group) == 1:
        bottom_right = RH_group
        top_right = RH_group
        max_slope = float('-inf')  # Start with the smallest possible slope
        min_slope = float('inf')   # Start with the largest possible slope
        # you need to maintain a clockwise ordering
        for point in LH_group:
            slope = get_slope(RH_group[0], point)
            if slope > max_slope:
                max_slope = slope
                top_left = point  # Keep track of the point that gives the largest slope
                    # Check for minimum slope
        if slope < min_slope:
            min_slope = slope
            bottom_left = point  # Keep track of the point with the smallest slope
        

    else:# get the coordinates that mark the new hulls
        top_left, top_right = find_upper_tang(LH_group, RH_group)
        bottom_left, bottom_right = find_lower_tang(LH_group, RH_group)
    top_tangent = [top_left, top_right]
    bottom_tangent = [bottom_right, bottom_left]
    # gett the index numbers of the points so we can merge into a hull
    LH_upper_tang = LH_group.index(top_left)
    RH_upper_tang = RH_group.index(top_right)

    LH_lower_tang = LH_group.index(bottom_left)
    RH_lower_tang = RH_group.index(bottom_right)

    hull_points = [] #initialize at leftmost point
    index = 0
    while 1: # put all the points from LH into list until the upper tang is found
        if index == LH_upper_tang:
            hull_points.append(top_left)
            break
        else:
            hull_points.append(LH_group[index])
        index = (index + 1) % len(LH_group)

    index = RH_upper_tang
    while 1:
        if index == RH_lower_tang:
            hull_points.append(bottom_right)
            break
        else:
            hull_points.append(RH_group[index])
        index = (index + 1) % len(RH_group)
    
    index = LH_lower_tang
    while 1:
        if index == 0: # you already added the first point
            break
        elif index == LH_upper_tang: # probably this should never happen
            break
        else:
            hull_points.append(LH_group[index])
        index = (index + 1) % len(LH_group)
    draw_hull(hull_points)
    plt.pause(.0005)
    return hull_points



# Assuming these functions are defined elsewhere
# from your_module import generate_random_points, compute_hull

# # List of point counts to test
# point_counts = [10, 100, 1000, 10000, 100000, 500000, 1000000]

# # Number of times to run the entire experiment
# num_iterations = 5

# # Color map for different colors in each iteration
# colors = plt.cm.viridis(np.linspace(0, 1, num_iterations))

# # Create a figure for plotting
# plt.figure(figsize=(10, 6))

# for iteration in range(num_iterations):
#     elapsed_times = []  # To store elapsed times for this iteration
    
#     for val in point_counts:
#         # Generate random points
#         points = generate_random_points('guassian', val, 312)
        
#         # Measure execution time
#         tic = time.time()
#         candidate_hull = compute_hull(points)
#         toc = time.time()
        
#         elapsed_time = toc - tic
#         elapsed_times.append(elapsed_time)
        
#         print(f"Iteration {iteration + 1}, Elapsed time for {val} points: {elapsed_time} seconds")
    
#     # Plot the results for this iteration with a unique color
#     plt.plot(point_counts, elapsed_times, marker='o', linestyle='-', color=colors[iteration], label=f'Iteration {iteration + 1}')

# # Add title and labels
# plt.title('Execution Time of compute_hull vs. Number of Points (5 Runs)')
# plt.xlabel('Number of Points')
# plt.ylabel('Elapsed Time (seconds)')

# # Add legend to show which color corresponds to which iteration
# plt.legend(title="Iterations")

# # Show grid and plot
# plt.grid(True)
# plt.show()


#their testing

points = generate_random_points('guassian', 500, 312)
tic = time.time()
candidate_hull = compute_hull(points)
toc = time.time()
elapsed_time = toc - tic
# Calculate the elapsed time
print(elapsed_time)
print(is_convex_hull(candidate_hull, points))
plt.figure()
plot_points(points)
draw_hull(candidate_hull)
plt.show()

# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.plot(point_counts, elapsed_times, marker='o', linestyle='-', color='b')
# plt.title('Execution Time of compute_hull vs. Number of Points')
# plt.xlabel('Number of Points')
# plt.ylabel('Elapsed Time (seconds)')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.grid(True)
# plt.show()

# # List of point counts to test
# point_counts = [10, 100, 1000, 10000, 100000, 500000, 1000000]


# for count in point_counts:
#     # Generate random points
#     points = generate_random_points('gaussian', count, 312)
    
#     # Measure execution time
#     tic = time.time()
#     candidate_hull = compute_hull(points)
#     toc = time.time()
    
#     elapsed_time = toc - tic

    
#     print(f"Elapsed time for {count} points: {elapsed_time} seconds")



















#my own testing
# np.random.seed(10) # 42 works
# random_pairs = np.random.uniform(-100, 100, size=(50, 2))

# random_tuples = [tuple(pair) for pair in random_pairs]
# compute_hull(random_tuples)

"""Lower Tangent
POI_right = left-most point Right-hand group
POI_left = right-most poin in Left-hand Group
old_slope = get_slope() # always do get slope with left point, right point

old_point = POI_right # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
for points in Right-hand group # now we want to go throughc ccw
    new_slope = get_slope(points, POI_Left)
    if new_slope => old_slope: # if the slope gets less positive doesnt change then we wanted that last point
        POI_right = old_point
        break # get out of here because you have the new right point
    old_slope = new_slope
    old_point = points

old_point = POI_left # you need to keep track of the old point b/c that's what will get stored once you've gone one too far
for points in left-hand group: # now go through clockwise
    new_slope = get_slope(points, POI_Right)
    if new_slope <= old_slope: # if the slope gets les negative doesnt change then we wanted that last point
        POI_left = old_point #once you take a new point you don't care about the slope to the last point
        break # get out of this for-loop because we have the next value in the left-hand group
    old_slope = new_slope
    old_point = points

# we need to stop testing these once the first step enters the if's
"""

"""cOMBINING THE HULLS AFTER FINDING THE TANGENTS
coordinates_of_upper = upper_tangent()"""