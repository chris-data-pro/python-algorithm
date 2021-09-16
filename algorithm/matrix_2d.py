from collections import deque


class Matrix2D:
    """
    2D Array implementations
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
    It is guaranteed target_map[0][0] = 0. There is only one target in the map.
    @param target_map: list of list
    @return: integer
    """

    """
    bfs
    """
    def shortest_path_2target_bfs(self, target_map):
        if not target_map or not target_map[0]:
            return -1  # to pass 1563

        return self.bfs(target_map, (0, 0))

    def bfs(self, matrix, node):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        points = deque([(node[0], node[1])])  # points only store value 0 or 2
        point_step = {(node[0], node[1]): 0}

        while points:
            curr = points.popleft()
            x, y = curr[0], curr[1]

            if matrix[x][y] == 2:
                return point_step[curr]

            for dx, dy in directions:
                newx, newy = x + dx, y + dy
                if (newx, newy) in point_step:
                    continue
                if 0 <= newx < len(matrix) and 0 <= newy < len(matrix[0]) and matrix[newx][newy] != 1:
                    point_step[(newx, newy)] = point_step[curr] + 1
                    points.append((newx, newy))
        return -1

    """
    dfs
    """
    def shortest_path_2target_dfs(self, target_map):
        if not target_map or not target_map[0]:
            return -1  # to pass 1563

        rows, cols = len(target_map), len(target_map[0])
        self.visited = set([(0, 0)])
        self.res = rows * cols

        self.dfs(target_map, 0, (0, 0))
        return -1 if self.res == rows * cols else self.res

    def dfs(self, matrix, steps, node):
        x, y = node[0], node[1]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for d_x, d_y in directions:
            n_x, n_y = x + d_x, y + d_y
            if 0 <= n_x < len(matrix) and 0 <= n_y < len(matrix[0]) and (n_x, n_y) not in self.visited:
                if matrix[n_x][n_y] == 2:
                    steps += 1
                    self.res = min(self.res, steps)
                    return
                elif matrix[n_x][n_y] == 1:
                    continue
                elif matrix[n_x][n_y] == 0:
                    steps += 1
                    self.visited.add((n_x, n_y))
                    self.dfs(matrix, steps, (n_x, n_y))
                    self.visited.remove((n_x, n_y))
                    steps -= 1
        return

    """
    0 - space, 1 - wall, 2 - target
    Find the maximum distance between any 0 to its nearest target 2.
    Starting from any point who's value is 0
    """
    def max_shortest_path_2target_dfs(self, input):
        if not input or not input[0]:
            return float('inf')  # by requirement

        rows, cols = len(input), len(input[0])
        max_distance = 0

        for i in range(rows):
            for j in range(cols):
                if input[i][j] == 0:
                    self.visited = set([(i, j)])
                    self.res = rows * cols
                    self.dfs(input, 0, (i, j))
                    max_distance = max(max_distance, float('inf') if self.res == rows * cols else self.res)

        return max_distance

    def max_shortest_path_2target_bfs(self, input):
        if not input or not input[0]:
            return float('inf')  # by requirement

        rows, cols = len(input), len(input[0])
        max_distance = 0

        for i in range(rows):
            for j in range(cols):
                if input[i][j] == 0:
                    shortest_2target_from_ij = self.bfs(input, (i, j))
                    max_distance = max(max_distance,
                                       float('inf') if shortest_2target_from_ij == -1 else shortest_2target_from_ij)
        return max_distance


if __name__ == '__main__':
    m2d = Matrix2D()
    print(m2d.spiral_order_idx(3, 4))
    print(m2d.max_rectangle([[1, 1, 0, 0, 1],
                             [0, 1, 0, 0, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 1, 1, 1],
                             [0, 0, 0, 0, 1]]))
    print('dfs:')
    print(m2d.shortest_path_2target_dfs([[0, 0, 0],
                                         [0, 0, 1],
                                         [0, 0, 2]]))
    print(m2d.max_shortest_path_2target_dfs([[1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                                             [1, 1, 2, 0, 0, 1, 1, 0, 0, 2],
                                             [2, 0, 0, 0, 1, 1, 2, 0, 1, 0]]))  # expect 5
    print(m2d.max_shortest_path_2target_dfs([[1, 1, 1, 1],
                                             [0, 0, 2, 0],
                                             [1, 1, 1, 1]]))  # expect 2
    print(m2d.max_shortest_path_2target_dfs([[1, 1, 1, 1],
                                             [1, 0, 0, 1],
                                             [1, 1, 1, 1]]))  # expect inf
    print(m2d.max_shortest_path_2target_dfs([[0, 0, 0],
                                             [0, 2, 0],
                                             [0, 0, 0]]))  # expect 2
    print('bfs:')
    print(m2d.shortest_path_2target_bfs([[0, 0, 0],
                                         [0, 0, 1],
                                         [0, 0, 2]]))
    print(m2d.max_shortest_path_2target_bfs([[1, 1, 1, 1, 0, 0, 0, 1, 1, 0],
                                             [1, 1, 2, 0, 0, 1, 1, 0, 0, 2],
                                             [2, 0, 0, 0, 1, 1, 2, 0, 1, 0]]))  # expect 5
    print(m2d.max_shortest_path_2target_bfs([[1, 1, 1, 1],
                                             [0, 0, 2, 0],
                                             [1, 1, 1, 1]]))  # expect 2
    print(m2d.max_shortest_path_2target_bfs([[1, 1, 1, 1],
                                             [1, 0, 0, 1],
                                             [1, 1, 1, 1]]))  # expect inf
    print(m2d.max_shortest_path_2target_bfs([[0, 0, 0],
                                             [0, 2, 0],
                                             [0, 0, 0]]))  # expect 2
