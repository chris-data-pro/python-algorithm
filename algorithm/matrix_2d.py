# Problems related to 2D matrix


class Matrix2D:
    """

    """

    """
    769 - 54
    Given an m x n matrix, 
    return all element's indices [i][j] of the matrix in spiral order
    @param rows: integer - number of rows
    @param cols: integer - number of columns
    @return: list of tuples - indices (i, j) in spiral order
    """
    def spiral_order_idx(self, rows, cols):
        if rows < 1 or cols < 1:
            return
        up = left = 0
        right = cols - 1
        down = rows - 1

        i, j = 0, 0
        res = [(i, j)]
        while len(res) < rows * cols:
            while j < right:
                j += 1
                res.append((i, j))
            while i < down:
                i += 1
                res.append((i, j))
            if up != down:
                while j > left:
                    j -= 1
                    res.append((i, j))
            if left != right:
                while i > up + 1:
                    i -= 1
                    res.append((i, j))

            left += 1
            right -= 1
            up += 1
            down -= 1

        return res

    """
    433
    Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. 
    If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.
    @param grid: list of list of integers - a boolean 2D matrix
    @return: integer - the number of islands
    """
    def num_islands_dfs(self, grid):
        return


if __name__ == '__main__':
    m2d = Matrix2D()
    print(m2d.spiral_order_idx(3, 4))
