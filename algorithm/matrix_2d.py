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
    def num_islands_bfs(self, grid):
        return

    """
    510
    Given a 2D boolean matrix filled with False and True, 
    find the largest rectangle containing all True and return its area.
    @param matrix: a boolean 2D matrix
    @return: an integer
    """
    def max_rectangle(self, matrix):
        def get_max_area(nums):
            nums += [-1]
            st = []
            ret = 0
            for i in range(len(nums)):
                while st and nums[i] <= nums[st[-1]]:
                    h = nums[st.pop()]
                    left_bound = st[-1] + 1 if st else 0
                    right_bound = i - 1
                    ret = max(ret, h * (right_bound - left_bound + 1))
                st.append(i)
            return ret

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i == 0:
                    matrix[i][j] = int(matrix[i][j])
                elif matrix[i - 1][j] >= 1 and matrix[i][j] == 1:
                    matrix[i][j] = matrix[i - 1][j] + 1
                else:
                    matrix[i][j] = int(matrix[i][j])

        max_area = 0
        for r in matrix:
            max_area = max(max_area, get_max_area(r))
        return max_area

    """
    1563
    Given a 2D boolean matrix filled with 0 - space, 1 - wall, 2 - target
    Starting from [0,0], find the shortest path that can reach target, and return the length of the path.
    You can only go up, down, left and right. 
    """
    """
    @param targetMap: 
    @return: nothing
    """
    def shortest_path_2target(self, targetMap):
        if not targetMap:
            return 0

        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.visited = set([(0, 0)])
        self.res = len(targetMap) * len(targetMap[0])

        self.dfs_shortest_path(targetMap, 0, (0, 0))
        return -1 if self.res == len(targetMap) * len(targetMap[0]) else self.res

    def dfs_shortest_path(self, targetMap, steps, node):
        x, y = node[0], node[1]
        for d_x, d_y in self.directions:
            n_x, n_y = x + d_x, y + d_y
            if 0 <= n_x < len(targetMap) and 0 <= n_y < len(targetMap[0]) and (n_x, n_y) not in self.visited:
                if targetMap[n_x][n_y] == 2:
                    steps += 1
                    self.res = min(self.res, steps)
                    return
                elif targetMap[n_x][n_y] == 1:
                    continue
                elif targetMap[n_x][n_y] == 0:
                    steps += 1
                    self.visited.add((n_x, n_y))
                    self.dfs_shortest_path(targetMap, steps, (n_x, n_y))
                    self.visited.remove((n_x, n_y))
                    steps -= 1
        return


if __name__ == '__main__':
    m2d = Matrix2D()
    print(m2d.spiral_order_idx(3, 4))
    print(m2d.max_rectangle([[1, 1, 0, 0, 1],
                             [0, 1, 0, 0, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 1]]))
