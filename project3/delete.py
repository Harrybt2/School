
def calculate_path(path_dict, source, target):
    current_key = target
    cost = path_dict[target][0]
    path_traveled = []
    while current_key != source:
        path_traveled.insert(0, current_key)
        current_key = path_dict[current_key][1]
        # print(current_key)
    path_traveled.insert(0, current_key)
    return path_traveled, cost




def find_shortest_path_with_heap(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    # make an array which shows where in the minheap objects are, the distance to that point and where you came from to get there
    pointer_array = {key: [float('inf'), None, None] for key in graph}# initialize the array with space 4V because I have 3 values for every key
    pointer_array[source] = [0, None, 0]# make the source's distance 0, location at front of min_heap
    # create the min heap with the starting value as the first one
    min_heap = []
    for keys, things in pointer_array.items():# space complexity of 2V because I add in two things from the pointer array
        min_heap.append([keys, things[0]])
        
    # while there are still elements in the heap
    while len(min_heap) != 0:
        node = min_heap[0]# Access the starting point and look at everything it goes to, pop it off the min heap and remember it
        node_of_interest = node[0]
        dist_to_node_of_interest = node[1]
        # need to change the pointer array so popping off a point in the min_heap doesn't ruin the pointer's indexing
        connections = graph[node_of_interest] # this gives us all the connections to a node
        for path, distance in connections.items():
            current_dist = pointer_array[path][0] # the current distance to this node
            if current_dist > distance + dist_to_node_of_interest : # Anytime the distance is less replace it and push it onto the min heap
                # This shouldnt happen now!! if its in the minheap just update it in the minheap, if not, through it in the back
                if pointer_array[path][2] == None:
                    min_heap.append([path, distance + dist_to_node_of_interest])
                    pointer_array[path]= [distance + dist_to_node_of_interest, node_of_interest, len(min_heap)-1] # update the pointer array with this nodes distance to get to it, previous node, and location in minheap
                else:
                    # find where in the minheap that point is and update the distance
                    loc_in_heap = pointer_array[path][2]
                    min_heap[loc_in_heap][1] = distance + dist_to_node_of_interest
                    pointer_array[path][0] = distance + dist_to_node_of_interest
                    pointer_array[path][1] = node_of_interest
                sift_up(min_heap, 0, pointer_array)# this is a part of the "append" in dijkstra's psuedocode, it will be at worst log(V)
        swap_pop(min_heap, pointer_array)
        sift_down(min_heap,0, pointer_array) # percolate
        
    nodes_traversed, cost = calculate_path(pointer_array,source, target)

    return nodes_traversed, cost

def create_heap(dict, starting_point, pointer):
    heap = []
    for key in dict: # set all the distances to infinity initially, with space for 'prev' arg
        heap.append([key, float('inf')])
    heap[starting_point] = [starting_point, 0] # set the distance to zero for your starting point
    heap = update_heap(heap, pointer)
    return heap

def make_pointers(graph):
    address = []
    n = 0
    for key in graph:
        address.append([key,n])
        n = n + 1 # increment the location
    return address

def update_heap(minheap, pointer):
    # look if any child is smaller than its parent
    # if yes, bubble it up the tree until its not bigger than its parent
    # if no leave it
    n = len(minheap)

    # Start from the last non-leaf node and sift_down all nodes in reverse order
    for i in range(n // 2 - 1, -1, -1):
        sift_down(minheap, n, i, pointer)

    return minheap
# Helper function to sift_down the subtree rooted at index i
# n is the size of the heap
def sift_down(minheap, i, node_positions):
    n = len(minheap)  # Get the size of the heap
    smallest = i  # Initialize the smallest as the current node index
    
    while True:
        left = 2 * i + 1  # Left child index
        right = 2 * i + 2  # Right child index

        # Check if the left child exists and is smaller than the current node
        if left < n and minheap[left][1] < minheap[smallest][1]:
            smallest = left

        # Check if the right child exists and is smaller than the current smallest node
        if right < n and minheap[right][1] < minheap[smallest][1]:
            smallest = right

        # If the smallest is not the current node, swap and continue sifting down
        if smallest != i:
            # Swap the current node with the smallest child
            minheap[i], minheap[smallest] = minheap[smallest], minheap[i]
            
            # Update the node positions in the dictionary
            node_positions[minheap[i][0]][2] = i  # Update new index of the node at position i
            node_positions[minheap[smallest][0]][2] = smallest  # Update new index of the node at position smallest
            
            # Move to the next position (smallest) and continue sifting down
            i = smallest
        else:
            # If no swaps are needed, stop the loop
            break

def sift_up(minheap, i, pointer):
    parent = (i - 1) // 2  # Parent index

    # Bubble up while i is not the root and the current node is smaller than its parent
    while i > 0 and minheap[i][1] < minheap[parent][1]:
        # Swap the current node with its parent
        minheap[i], minheap[parent] = minheap[parent], minheap[i]
        
        # Update the pointer array to reflect the change in positions
        pointer[minheap[i][0]][2], pointer[minheap[parent][0]][2] = (
            pointer[minheap[parent][0]][1], pointer[minheap[i][0]][1]
        )

        # Move upwards
        i = parent
        parent = (i - 1) // 2
def swap_pop(minheap, pointer):
    last_place = len(minheap) -1
    pointer[minheap[-1][0]][2] = 0
    pointer[minheap[0][0]][2] = None
    minheap[0], minheap[last_place] = minheap[last_place], minheap[0] # swap
    
    minheap.pop() # pop off last one


def find_shortest_path_with_array(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    #p has space complexity 3V, 1 for each vertex key, 2 for the two things in the value
    p = {key: (float('inf'), None) for key in graph} # makes a dictionary in the form {node: (distance to it, node you came from)}
    p[source] = (0, None) # make the source's distance 0

    H = {source: p[source][0]} # initialize this ditionary with the source and the distance to it, has the form {node: diistance to it}
    # at worst this will be V*2 because i'm storing 2 values everytime I find a node, and worst case I find all of them
    print(H)
    while len(H) != 0:
        minimum_len_key= min(H, key=H.get) #find the key of the lowest value
    
        H.pop(minimum_len_key) # remove the lowest fromt the priority queue
        connections = graph[minimum_len_key] # get all the connections from that node
        for path, distance in connections.items(): # look at where the node goes to and the distance to them
            current_dist = p[path][0]
            if  current_dist > distance + p[minimum_len_key][0]: # if it costs less to get there replace the distances
                p[path] = (distance + p[minimum_len_key][0], minimum_len_key) 
                H[path] = distance + p[minimum_len_key][0]
    # print('18: ' + str(p[301]) + ', 380: ' + str(p[936]))
    # print(p)
    list_of_nodes_traveled, cost_of_path = calculate_path(p, source, target) # get the stuff to return
    
    return list_of_nodes_traveled, cost_of_path


graph = {
        0: {1: 2, 2: 1, 3: 4},
        1: {0: 2, 2: 1, 3: 1},
        2: {0: 4, 1: 4},
        3: {0: 3, 1: 2, 2: 1}
}
print(find_shortest_path_with_heap(graph, 0, 3))


