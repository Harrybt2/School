""" Initiate Pointer array containg all nodes and the cost to get there (zero for starting point, inf for teh rest)
    Begin at a starting node, look at everything it connects to
    store those distances in your min-heap
    perculate the min heap
    update the pointer array with any new distance smaller than the last and their index in the min heap
    swap, pop and perculate the min heap
    search at the next node (the lowest)
You have a pointer array that stores the nodes 
"""




def find_shortest_path_with_heap(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the heap-based algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """


def find_shortest_path_with_array(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the array-based (linear lookup) algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """
