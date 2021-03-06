from collections import deque


class Matrix2D:
    """
    2D Array implementations
    """

    """
    769 - 54
    Given an m x n matrix, 
    return all element's indices [i][j] of the matrix in spiral order
    
    Input: 3, 3
    Output: [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (1, 1)]  右下左上 - 右下左上 - ...
    
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
    
    输入：
    [
      [1,1,0,0,0],
      [0,1,0,0,1],
      [0,0,0,1,1],
      [0,0,0,0,0],
      [0,0,0,0,1]
    ]
    输出：
    3
    
    @param grid: list of list of integers - a boolean 2D matrix
    @return: integer - the number of islands
    """
    def num_islands_bfs(self, grid):
        if not grid or not grid[0]:
            return 0

        islands = 0
        self.visited_set = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] and (i, j) not in self.visited_set:
                    self.bfs_ni(grid, i, j)  # 每次bfs处理完一个岛屿里的所有node
                    islands += 1

        return islands

    def bfs_ni(self, grid, x, y):
        queue = deque([(x, y)])
        self.visited_set.add((x, y))
        while queue:
            x, y = queue.popleft()
            for delta_x, delta_y in [(1, 0), (0, -1), (-1, 0), (0, 1)]:
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.is_valid_ni(grid, next_x, next_y) or grid[next_x][next_y] == 0:
                    continue
                queue.append((next_x, next_y))
                self.visited_set.add((next_x, next_y))

    def is_valid_ni(self, grid, x, y):
        n, m = len(grid), len(grid[0])
        if not (0 <= x < n and 0 <= y < m):
            return False
        if (x, y) in self.visited_set:
            return False
        return True  # return grid[x][y]: 1 - True (valid), or 0 - False (invalid)

    """
    510
    Given a 2D boolean matrix filled with False and True, 
    find the largest rectangle containing all True and return its area.
    
    Input:
    [
      [1, 1, 0, 0, 1],
      [0, 1, 0, 0, 1],
      [0, 0, 1, 1, 1],
      [0, 0, 1, 1, 1],
      [0, 0, 0, 0, 1]
    ]
    Output: 6  最大长方形的面积6
    
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
    
    Input:
    [
     [0, 0, 0],
     [0, 0, 1],
     [0, 0, 2]
    ]
    Output: 4
    Explanation: [0,0] -> [1,0] -> [2,0] -> [2,1] -> [2,2]
    
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

    def bfs(self, matrix, node):  # 返回node到最近2的最短距离
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

    def dfs(self, matrix, steps, node):  # 会修改外部global self.res
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
    for each O, there might exists a shortest distance to its nearest G, otherwise it's infinity. 
    among all these shortest distance, find the max
    Starting from any point who's value is 0
    
    XXXXOOOXXO
    XXGOOXXOOG maximum distance is 5
    GOOOXXGOXO 
    
    OOO
    OGO  max distance = 2
    OOO

    bfs is the better solution
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
                    self.res = rows * cols  # 对每一个0都找到最短距离
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

    """
    33
    N Queens Puzzle
    placing n queens on an n×n chessboard, Any two queens can't be in the same row, same column, same diagonal line 斜线
    Given an integer n, return all distinct solutions to the N-queens puzzle
    
    Input: n = 4
    Output:
    [
      // Solution 1
      [". Q . .",
       ". . . Q",
       "Q . . .",
       ". . Q ."
      ],
      // Solution 2
      [". . Q .",
       "Q . . .",
       ". . . Q",
       ". Q . ."
      ]
    ]
    
    """
    def n_queens_solutions(self, n):
        boards = []
        visited = {
            'col': set(),
            'sum': set(),  # row + col: /  sum相同的所有点都在同一斜线 / 上
            'diff': set(),  # row - col: \  diff相同的所有点都在同一斜线 \ 上
        }
        self.dfs_nqueens(n, [], visited, boards)
        return boards

    def dfs_nqueens(self, n, permutation, visited, boards):
        if n == len(permutation):
            boards.append(self.draw(permutation))
            return

        row = len(permutation)
        for col in range(n):
            if not self.is_valid(permutation, visited, col):
                continue
            permutation.append(col)
            visited['col'].add(col)
            visited['sum'].add(row + col)
            visited['diff'].add(row - col)
            self.dfs_nqueens(n, permutation, visited, boards)
            visited['col'].remove(col)
            visited['sum'].remove(row + col)
            visited['diff'].remove(row - col)
            permutation.pop()

    # O(1)
    def is_valid(self, permutation, visited, col):
        row = len(permutation)
        if col in visited['col']:
            return False
        if row + col in visited['sum']:
            return False
        if row - col in visited['diff']:
            return False
        return True

    def draw(self, permutation):
        board = []
        n = len(permutation)
        for col in permutation:
            row_string = ''.join(['Q' if c == col else '.' for c in range(n)])
            board.append(row_string)
        return board

    """
    34
    return the total number of distinct solutions.
    """
    def n_queens_total(self, n):
        li = [str(i) for i in range(n)]
        self.result = 0
        self.dfs_nqueens_total("", li)
        return self.result

    def dfs_nqueens_total(self, path, nums):
        if not nums:
            self.result += 1
            return
        for i in range(len(nums)):
            if not self.valid(path, nums[i]):
                continue
            self.dfs_nqueens_total(path+nums[i], nums[:i]+nums[i+1:])

    def valid(self, path, num):
        for i in range(len(path)):
            if abs(int(num) - int(path[i])) == abs(len(path)-i):
                return False
        return True

    """
    477
    Given a 2D board containing 'X' and 'O'
    flip all 'O''s into 'X''s in all regions surrounded by 'X'
    Input:
      X X X X
      X O O X
      X X O X
      X O X X
    Output:
      X X X X
      X X X X
      X X X X
      X O X X
    """
    def surrounded_regions_bfs(self, board):
        if not board or not board[0]:
            return

        rows, cols = len(board), len(board[0])
        self.visited = {}

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if board[i][j] == 'O' and (i, j) not in self.visited:
                    self.bfs_surrounded(board, (i, j))
        return

    def bfs_surrounded(self, matrix, node):
        points = deque([node])  # only 'O's
        self.visited[node] = 1
        group = {node: 1}
        valid = True

        while points:
            x, y = points.popleft()

            if x == 0 or x == len(matrix) - 1 or y == 0 or y == len(matrix[0]) - 1:
                valid = False

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                newx, newy = x + dx, y + dy
                if (newx, newy) in group:
                    continue
                if 0 <= newx < len(matrix) and 0 <= newy < len(matrix[0]) and matrix[newx][newy] == 'O':
                    points.append((newx, newy))
                    group[(newx, newy)] = 1
                    self.visited[(newx, newy)] = 1

        if valid:
            for i, j in group.keys():
                matrix[i][j] = 'X'

    """
    bfs in one
    """
    def surroundedRegions_bfs(self, board):
        if not board or not board[0]:
            return

        rows, cols = len(board), len(board[0])
        points = deque([])
        visited, group = {}, {}
        change = []
        valid = True

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if board[i][j] == 'O' and (i, j) not in visited:
                    points.append((i, j))  # only 'O's
                    group[(i, j)] = 1
                    visited[(i, j)] = 1

                    while points:
                        curr = points.popleft()

                        x, y = curr[0], curr[1]

                        if x == 0 or x == len(board) - 1 or y == 0 or y == len(board[0]) - 1:
                            valid = False

                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            newx, newy = x + dx, y + dy
                            if (newx, newy) in group:
                                continue
                            if 0 <= newx < len(board) and 0 <= newy < len(board[0]) and board[newx][newy] == 'O':
                                points.append((newx, newy))
                                group[(newx, newy)] = 1
                                visited[(newx, newy)] = 1
                    if valid:
                        change += list(group.keys())
                    else:
                        group = {}

                    valid = True

        for x, y in change:
            board[x][y] = 'X'

        return

    """
    1205
    Given a matrix of M x N elements (M rows, N columns), 
    return all elements of the matrix in diagonal order as shown in the following example.
    
    Input:
    [
        [ 1, 2, 3 ],
        [ 4, 5, 6 ],
        [ 7, 8, 9 ]
    ]
    Output:
    [1,2,4,7,5,3,6,8,9]   /向右上 (sum(i, j) == 0), /向左下 (sum(i, j) == 1), /向右上 (sum(i, j) == 2), /向左下 ...

    @param matrix: a 2D array
    @return: return a list of integers
    """
    def find_diagonal_order(self, matrix):
        rows, cols = len(matrix), len(matrix[0])

        ans = [[]] * (rows + cols - 1)  # ans里每个list都reference to the same object
        # ans = [[] for _ in range(rows + cols - 1)]  # ans里每个list都reference to不同的object，下面可以用 += 或 append
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                if (i + j) % 2:
                    # print(i, j, ans)
                    ans[i + j] = ans[i + j] + [matrix[i][j]]  # 不能用 ans[i + j] += [matrix[i][j]] 或 ans[i + j].append
                    # print(i, j, ans)
                else:
                    ans[i + j] = [matrix[i][j]] + ans[i + j]
        return [x for y in ans for x in y]

    def find_diagonal_order_bf(self, matrix):
        rows, cols = len(matrix), len(matrix[0])

        ans = []
        for rc_sum in range(rows + cols - 1):
            if rc_sum % 2:
                for i in range(0, rows, 1):
                    for j in range(0, cols, 1):
                        if i + j == rc_sum:
                            ans.append(matrix[i][j])
            else:
                for i in range(rows - 1, -1, -1):
                    for j in range(cols - 1, -1, -1):
                        if i + j == rc_sum:
                            ans.append(matrix[i][j])
        return ans


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
    print(m2d.find_diagonal_order([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]]))
