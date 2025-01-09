# This is for finding the contiguos subsequence of max sum of a 1D array
# Harrison Denning Oct 28, 2024

"""This algorithm is O(n) because you only go through the list one time, there is no recursion. all other operations of list lookups or appends or variable/ list 
creation are o(1)"""

# Input: an array
def contig_sub_max(num_list):
    # initialize everything starting at the first value in the list
    running_tot = num_list[0]
    running_range = [0]
    best_range = running_range
    best_tot = running_tot
    for indx in range(1,len(num_list)): # go through every position in the list, excluding the fisrt because our setup takes care of that
        # decide if you're better off including the next value in your sum or starting over
        if running_tot + num_list[indx] > num_list[indx]: # if its better to include it, update your running total and range
            running_tot = running_tot + num_list[indx]
            running_range.append(indx)
        else: # otherwise restart both
            running_tot = num_list[indx]
            running_range = [indx]
        if running_tot > best_tot: # update your running total
            best_tot = running_tot
            best_range = running_range
            ''' a note: you may have a case where the last value in your best option is negative, if that's the case this section
            will maintain a correct best sum, but the range over which this occurs will need to be corrected to exclude the last
            subtraction'''
    if num_list[best_range[-1]] < 0: # check if the last value added was negative, if it was, then remove it from the list
        best_range.pop() 
    if best_tot < 0:
        # if all the numbers happen to be negative, then return that the best total is 0 if you go a distance of 0 through the list
        return 'N/A'
    # okay now the problem acutally asks for the subsequence where this occurs, luckily lookups are constant time, so just convert the best range into the list values
    contig_subsequence = []
    for vals in best_range:
        contig_subsequence.append(num_list[vals])
    return contig_subsequence # returns the contiguous subsequence where this occurs    

# practice_array = [5,15,-30,10,-5,40,10]
practice_array = [10,100,-200,-3,-4]

print(contig_sub_max(practice_array))