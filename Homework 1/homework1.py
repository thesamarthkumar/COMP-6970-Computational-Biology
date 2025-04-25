'''
Samarth Kumar
COMP 6970
'''

import numpy as np
import pandas as pd
import scipy
from plotnine import *
'''
Homework 1 problem 8 -- global alignment
use the simple scoring method of +1 for match and -1 for mismatch and indel
print the global alignment with one line per string with '-' characters for indels
'''
def global_alignment(sequence1, sequence2):
    # Initialize the grid
    m, n = len(sequence1), len(sequence2)
    grid = np.full((m+1, n+1), 0)

    # Base Case
    for i in range(1, m+1):
        grid[i][0] = grid[i-1][0] - 1
    for i in range(1, n+1):
        grid[0][i] = grid[0][i-1] - 1

    # Calculate the score matrix.
    for i in range(1, m+1):
        for j in range(1, n+1):
            # Determine possible reward value for the current cell.
            if sequence1[i-1] == sequence2[j-1]:
                score = 1 # match
            else:
                score = -1 # mismatch or indel

            # Calculate the scores from the diagonal, horizontal, vertical.
            diagonal = grid[i-1][j-1] + score
            horizontal = grid[i-1][j] - 1
            vertical = grid[i][j-1] - 1

            # Assign the current value to be the max value out of the diagonal, horizontal, vertical.
            grid[i][j] = max(diagonal, horizontal, vertical)
    
    # Comment/uncomment the line below to hide/view the matrix.
    print(f'Score Matrix: \n{grid}\n')

    # Retrieve and print the global alignment using the helper function.
    a1, a2 = backtrace(grid, sequence1, sequence2)
    print(f'Global Alignment: \nSequence 1: {a1}\nSequence 2: {a2}')
    
    # Comment/uncomment the line below to hide/view the optimal alignment score.
    print(f'Optimal Alignment Score: {grid[m][n]}\n')


'''
Helper function for the backtracing
'''
def backtrace(grid, sequence1, sequence2):
    i, j = len(sequence1), len(sequence2)
    aligned1 = []
    aligned2 = []

    while i > 0 or j > 0:
        # Characters match or mismatch based on diagonal.
        if i > 0 and j > 0 and (grid[i][j] == 
                                grid[i-1][j-1] + (1 if sequence1[i-1] == sequence2[j-1] else -1)):
            aligned1.append(sequence1[i-1])
            aligned2.append(sequence2[j-1])
            i -= 1
            j -= 1

        # Handle gaps in either sequence1 or sequence2 by checking horizontal and vertical.
        elif i > 0 and grid[i][j] == grid[i-1][j] - 1:
            aligned2.append('-')
            aligned1.append(sequence1[i-1])
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(sequence2[j-1])
            j -= 1

    aligned1, aligned2 = ''.join(reversed(aligned1)), ''.join(reversed(aligned2))
    
    return aligned1, aligned2


'''
support code for creating random sequence, no need to edit
'''
def random_sequence(n):
    return("".join(np.random.choice(["A","C","G","T"], n)))


'''
support code for mutating a sequence, no need to edit
'''
def mutate(s, snp_rate, indel_rate):
    x = [c for c in s]
    i = 0
    while i < len(x):
        if np.random.random() < snp_rate:
            x[i] = random_sequence(1)
        if np.random.random() < indel_rate:
            length = np.random.geometric(0.5)
            if np.random.random() < 0.5: # insertion
                x[i] = x[i] + random_sequence(length)
            else:
                for j in range(i,i+length):
                    if j < len(x):
                        x[j] = ""
                        i += 1
        i += 1
    return("".join(x))

# creating related sequences
s1 = random_sequence(100)
s2 = mutate(s1, 0.1, 0.1)
# running your alignment code
global_alignment(s1, s2)
