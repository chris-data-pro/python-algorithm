"""
Sudoku Puzzle
Solve it by filling the empty cells
"""

if __name__ == '__main__':
    puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]

    def udf1(puzzle, row, col, digit):
        for j in range(9):
            if puzzle[row][j] == digit:
                return False
        for i in range(9):
            if puzzle[i][col] == digit:
                return False
        for i in range(3):
            for j in range(3):
                if puzzle[row//3 * 3 + i][col//3 * 3 + j] == digit:
                    return False
        return True

    def udf2(puzzle, i, j):
        m, n = 9, 9
        if j == n:
            return udf2(puzzle, i + 1, 0)
        if i == m:
            return True
        if puzzle[i][j] != 0:
            return udf2(puzzle, i, j + 1)

        for val in range(1, 10):
            if not udf1(puzzle, i, j, val):
                continue
            puzzle[i][j] = val
            if udf2(puzzle, i, j + 1):
                return True
            puzzle[i][j] = 0

        return False

    def udf3(puzzle):
        udf2(puzzle, 0, 0)
        return puzzle

    # print(udf3(puzzle))
    expected = [
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ]

    assert udf3(puzzle) == expected
    # print(puzzle)
