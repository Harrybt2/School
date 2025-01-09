import math

def does_cell_exist(key, matrix): # you must determine if its inbounds, if its not just return an infinity
    if key in matrix:
        return matrix[key][0]
    else: 
        return math.inf

def align(
        seq1: str,
        seq2: str,
        match_award=-3,
        indel_penalty=5,
        sub_penalty=1,
        banded_width=-1,
        gap='-'
) -> tuple[float, str | None, str | None]:
    """
        Align seq1 against seq2 using Needleman-Wunsch
        Put seq1 on left (j) and seq2 on top (i)
        => matrix[i][j]
        :param seq1: the first sequence to align; should be on the "left" of the matrix
        :param seq2: the second sequence to align; should be on the "top" of the matrix
        :param match_award: how many points to award a match
        :param indel_penalty: how many points to award a gap in either sequence
        :param sub_penalty: how many points to award a substitution
        :param banded_width: banded_width * 2 + 1 is the width of the banded alignment; -1 indicates full alignment
        :param gap: the character to use to represent gaps in the alignment strings
        :return: alignment cost, alignment 1, alignment 2
    """
    ## below does it with a list
    # E = [[0 for _ in range(len(seq1))] for _ in range(len(seq2))] # the table where answers will be filled in


    # for i in range(len(seq1)):
    #     E[i][0] = i*indel_penalty
    # for j in range(len(seq2)):
    #     E[0][j] = j*indel_penalty
    # i = 0
    # j = 0
    # for i in range(len(seq1)):
    #     for j in range(len(seq2)):
    #         if seq1[i] == seq2[j]:
    #             diff = match_award
    #         else:
    #             diff = sub_penalty
    #         E[i][j] = min(E[i-1][j] + indel_penalty, E[i][j- 1] + indel_penalty, E[i - 1][j - 1] + diff)
    # return E
    E = {} # create the empty dictionary
    seq1 = ' ' + seq1 # add an empty space onto both strings to make the matrix with the appropriate 0 row
    seq2 = ' ' + seq2
   
    for i in range(len(seq1)): # go through every letter in string 1
        if banded_width == -1: # if its a banded solution you want then only search the bandwidth
            go_up_to = len(seq2)
            upto = range(go_up_to)
        else: # otherwise look at it all
            start = max(0, i - banded_width)
            end = min(len(seq2), i + banded_width +1)
            upto = range(start,end)

        # look at if they wanted it banded, if so then you only go from j to + or - bandwidth

        for j in upto: # look at all the letters you've decided to and fill it in accordingly
            # this means we have O(n*m) space and time for non-banded and O(k*n) for banded
            # everything below this is just look-ups, refrences, addition and subtraction, which dont add anything above O(n) to the complexity
            # a few special cases to consider
            if i== 0 and j ==0:
                E[(i,j)] = (0,None)
            elif i == 0:
                E[(i,j)] = (j*indel_penalty,(i, j-1)) 
        
            elif j ==0:
                E[(i,0)] = (i*indel_penalty,(i-1, j))
            
            else:
                if seq1[i] == seq2[j]:
                    diff = match_award
                else:
                    diff = sub_penalty
                # determine what penalty most be applied as you move throught the matrix
                diag = (does_cell_exist((i - 1, j - 1), E) + diff, (i - 1, j- 1))
                left = (does_cell_exist((i, j-1), E) + indel_penalty, (i, j - 1))
                top = (does_cell_exist((i - 1, j), E) + indel_penalty, (i - 1, j))
                E[i,j] = min(diag, left, top, key = lambda t: t[0])

    value = E[(i,j)][0]
    
    # to make alignment strings, if you go diagonally just add the letter to both
    # if going up, letter in string one (left) and a gap in string two (right)
    # if going left its the opposite
    i = len(seq1)-1
    j = len(seq2)-1
    last = (i,j)
    alignment1 = ''
    alignment2 = ''

    # this next section just gets the alignments, this does not contribute significantly to the time complexity because at worst its O(n+m)
    # the alignment strings wont be larger than n + m either so that doesn;t contribute significantly tot he space complexity
    while last != (0,0):
        if E[i,j][1] == (i-1, j-1): # if you go diagonally just add the letter to both
            alignment1 += seq1[i] # these should add the same thing if matched
            alignment2 += seq2[j]
            # now go to that point
            i = i-1
            j = j-1
            last = (i,j)
        if E[i,j][1] == (i, j-1): # if you went left  string1 has a gap, string2 gets its letter
            
            alignment1 += gap
            alignment2 += seq2[j]

            # now go to that point
            j = j-1             
            last = (i,j)

                    
        if E[i,j][1] == (i - 1, j): # if you go up letter in string one (left) and a gap in string two (right)
            alignment1 += seq1[i]
            alignment2 += gap
            # now go to that point
            i = i-1 
            last = (i,j)

    return value, alignment1[::-1], alignment2[::-1] 




# print(align('polynomial', 'exponential'))
# print(align('GGGGTTTTAAAACCCCTTTT', 'TTTTAAAACCCCTTTTGGGG', banded_width=2))



"""
psuedocode from the book:

for i =0,1,2,...,m: # go through every letter in string m and make the first row of your table
 E(i,0) = i
 for j =1,2,...,n: # go through every letter in string n and make 
 E(0,j) = j 
 for i =1,2,...,m:
 for j =1,2,...,n:
 E(i,j) = min{E(i −1,j) +1,E(i,j −1)+1,E(i −1,j −1)+diff(i,j)}
 return E(m,n

 """
