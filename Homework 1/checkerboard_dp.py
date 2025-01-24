'''
Given a matrix of 0s and 1s, return the size of the largest checkerboard
pattern that can be formed. A checkerboard is formed by alternating 0s and 1s and
must be square.
'''

def largest_checkerboard(matrix):
    m, n = len(matrix), len(matrix[0])
    size_matrix = [[0] * n for _ in range(m)]
    maxSize = 0

    # Fill size_matrix table
    for i in range(m):
        
        for j in range(n):

            # Base case. First row and first column
            if i == 0 or j == 0:  
                size_matrix[i][j] = 1
            else:
                # Check alternating pattern with neighbors. Adjacent cells must be different.
                condition = (
                    (matrix[i][j] != matrix[i-1][j]) and
                    (matrix[i][j] != matrix[i][j-1]) and
                    (matrix[i][j] == matrix[i-1][j-1])
                )
                if (condition):
                    # Take the minimum value from top, left, and diagonal adjacent cells.
                    min_score = min(
                        size_matrix[i-1][j],
                        size_matrix[i][j-1],
                        size_matrix[i-1][j-1]
                    )
                    # Update the matrix with the above minimum value + 1.
                    size_matrix[i][j] = 1 + min_score
                else:
                    # size matrix is 1 if the above is not met, indicating a 1x1 checkerboard instead.
                    size_matrix[i][j] = 1
            
            # Update maximum size
            maxSize = max(maxSize, size_matrix[i][j])

    # Print the size_matrix table
    print("size_matrix Array:")
    for row in size_matrix:
        print(row)

    return maxSize

# Test the function with sample grids

grid = [
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 1]
]

# Returns 3
grid2 = [
    [0,1,1,1,0,1],
    [1,0,1,0,1,0],
    [0,1,0,1,0,1],
    [1,0,1,0,1,0],
    [1,0,0,1,1,0]
]

# Returns 4
match = [
    [0,1,0,1],
    [1,0,1,0],
    [0,1,0,1],
    [1,0,1,0]
]

s = largest_checkerboard(grid)
print(f'Largest Checkerboard Size: {s}')