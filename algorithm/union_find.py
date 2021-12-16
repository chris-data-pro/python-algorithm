# Write a function usinng union find


class UnionFind:
    def __init__(self):
        self.count = 0  # initially 0 disconnected points
        self.parent = {}  # element -> parent element (element can't be Point(x, y) objects, can be (x, y) instead)
        self.weight = {}  # element -> int (size of the same group)

    def add(self, item):
        if item in self.parent:  # first check if it's already in
            return
        self.parent[item] = item  # initially each point's parent is itself
        self.weight[item] = 1  # initially each tree has 1 point
        self.count += 1

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:  # first check if they are already connected
            return

        # to form a balanced tree, so that tree height will be around logN
        if self.weight[rootP] > self.weight[rootQ]:
            self.parent[rootQ] = rootP
            self.weight[rootP] += self.weight[rootQ]
        else:
            self.parent[rootP] = rootQ
            self.weight[rootQ] += self.weight[rootP]

        self.count -= 1

    def connected(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)

        return rootP == rootQ

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # connect to its grandparent, also shorten the tree
            x = self.parent[x]  # become its grandparent

        return x

    """
    433
    Given a boolean 2D matrix, 0 is represented as the sea, 1 is represented as the island. 
    If two 1 is adjacent, we consider them in the same island. We only consider up/down/left/right adjacent.
    
    Input: 
    [[1,1,0,0,0],
     [0,1,0,0,1],
     [0,0,0,1,1],
     [0,0,0,0,0],
     [0,0,0,0,1]]
    Output: 3
    
    @param grid: list of list of integers - a boolean 2D matrix
    @return: integer - the number of islands
    """
    def num_islands_uf(self, grid):
        if not grid or not grid[0]:
            return 0
        uf = UnionFind()
        rows, cols = len(grid), len(grid[0])
        # lower, right, upper, left
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # (-1, -1) upper left, (-1, 1) upper right, etc

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    uf.add((i, j))
                    for dx, dy in directions:
                        newx, newy = i + dx, j + dy
                        if 0 <= newx < rows and 0 <= newy < cols and grid[newx][newy] == 1:
                            uf.add((newx, newy))  # uf.add will check if (newx, newy) is already in before adding
                            uf.union((i, j), (newx, newy))
        return uf.count

    """
    677
    Find the number of islands that size bigger or equal than k
    
    Input: 
    [[1,1,0,0,0],
     [0,1,0,0,1],
     [0,0,0,1,1],
     [0,0,0,0,0],
     [0,0,0,0,1]]
    2
    Output: 2
    
    @param grid: a 2d boolean array
    @param k: an integer
    @return: the number of Islands with size >= k
    """
    def num_islands_k_more(self, grid, k):
        if not grid or not grid[0]:
            return 0
        ufnik = UnionFind()
        rows, cols = len(grid), len(grid[0])

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ufnik.add((i, j))
                    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        newx, newy = i + dx, j + dy
                        if 0 <= newx < rows and 0 <= newy < cols and grid[newx][newy] == 1:
                            ufnik.add((newx, newy))  # uf.add will check if (newx, newy) is already in before adding
                            ufnik.union((i, j), (newx, newy))

        roots = [item[0] for item in ufnik.parent.items() if item[0] == item[1]]  # get all roots
        count = 0
        for root in roots:
            if ufnik.weight[root] >= k:  # weight of each root
                count += 1
        return count

    """
    860
    Count the number of DISTINCT islands. 
    An island is considered to be the same as another if and only if one island has the same shape as another island
    Do NOT consider rotated or reflected 
    
    Input:
      [
        [1,1,0,0,0],
        [0,1,0,0,0],
        [1,0,0,1,1],
        [1,1,0,0,1]
      ]
    Output: 2 三个islands里有两个形状一样
    """
    def num_distinct_islands(self, grid):
        if not grid or not grid[0]:
            return 0
        ufd = UnionFind()
        rows, cols = len(grid), len(grid[0])

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ufd.add((i, j))
                    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        newx, newy = i + dx, j + dy
                        if 0 <= newx < rows and 0 <= newy < cols and grid[newx][newy] == 1:
                            ufd.add((newx, newy))  # uf.add will check if (newx, newy) is already in before adding
                            ufd.union((i, j), (newx, newy))

        islands = {}  # root -> [all nodes in this island]
        for key in ufd.parent:
            root = ufd.find(key)
            islands[root] = islands.get(root, []) + [key]

        shapes = set()
        for island in islands.values():  # each island is a list of nodes
            shapes.add(self.move_tl(island))  # 使用tuple或者string记录相对路径!

        return len(shapes)

    """
    move island (a list of coordinates) to top left corner
    """
    def move_tl(self, shape):
        shape.sort(key=lambda v: (v[0], v[1]))  # shape = sorted(shape)
        min_x, min_y = shape[0]
        # return [(i - min_x, j - min_y) for i, j in shape]  # error when add to set: unhashable type: 'list'
        return ",".join('(' + str(i - min_x) + ',' + str(j - min_y) + ')' for i, j in shape)  # string

    """
    804
    Count the number of DISTINCT islands. 
    An island is considered to be the same as another if and only if one island has the same shape as another island
    Do consider rotated or reflected 
    
    Input:
      [
        [1,1,1,0,0],
        [0,0,1,0,0],
        [0,0,0,1,0],
        [0,1,1,1,0]
      ]
    Output: 1  两个islands翻转或对调后形状一样
    """
    def num_distinct_islands_rotate_flip(self, grid):
        if not grid or not grid[0]:
            return 0
        ufdrf = UnionFind()
        rows, cols = len(grid), len(grid[0])

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    ufdrf.add((i, j))
                    for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                        newx, newy = i + dx, j + dy
                        if 0 <= newx < rows and 0 <= newy < cols and grid[newx][newy] == 1:
                            ufdrf.add((newx, newy))  # uf.add will check if (newx, newy) is already in before adding
                            ufdrf.union((i, j), (newx, newy))

        islands = {}  # root -> [all nodes in this island]
        for key in ufdrf.parent:
            root = ufdrf.find(key)
            islands[root] = islands.get(root, []) + [key]

        shapes = set()
        for island in islands.values():  # each island is a list of nodes
            same_shape_islands = self.canonical(island)
            island_represent = min([self.move_tl(ssi) for ssi in same_shape_islands])
            shapes.add(island_represent)  # 使用tuple或者string记录相对路径!

        return len(shapes)

    """
    rotate and flip， return the total 8 shapes for the input shape
    
    input: [(0,0),(0,1),(0,2),(1,2)]                 *  *  *
                                                           *
                                                           
    output: [[(0, 0), (0, 1), (0, 2), (1, 2)],                     # self
    
             [(0, 0), (1, 0), (2, 0), (2, -1)],         *
                                                        *
                                                     *  *
                                                 
             [(0, 0), (0, -1), (0, -2), (-1, -2)],   *
                                                     *  *  *
                                                     
             [(0, 0), (-1, 0), (-2, 0), (-2, 1)],    *  *
                                                     *
                                                     *
                                                     
             [(0, 0), (0, 1), (0, 2), (-1, 2)],            *       # up down reflection of self 
                                                     *  *  *
             
             [(0, 0), (1, 0), (2, 0), (2, 1)],       *
                                                     *
                                                     *  *
             
             [(0, 0), (0, -1), (0, -2), (1, -2)],    *  *  *       # left right reflection
                                                     *
             
             [(0, 0), (-1, 0), (-2, 0), (-2, -1)]]   *  *
                                                        *
                                                        *
    
    @param shape: list of coordinates
    @return: list of 8 elements, each element being a shape (list of coordinates)
    """
    def canonical(self, shape):
        # shapes = [[(a * i, b * j) for i, j in shape] for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        # shapes += [[(j, i) for i, j in shape] for shape in shapes]
        shape = [(i, j) for i, j in shape]
        shape_cw_90 = [(j, -i) for i, j in shape]
        shape_cw_180 = [(-i, -j) for i, j in shape]
        shape_cw_270 = [(-j, i) for i, j in shape]
        rotate = [shape, shape_cw_90, shape_cw_180, shape_cw_270]

        flip = [(-i, j) for i, j in shape]  # up side down flip
        flip_cw_90 = [(j, i) for i, j in shape]
        flip_cw_180 = [(i, -j) for i, j in shape]
        flip_cw_270 = [(-j, -i) for i, j in shape]
        flip = [flip, flip_cw_90, flip_cw_180, flip_cw_270]

        shapes = rotate + flip

        return shapes

    """
    434
    Originally, the 2D matrix is all 0 which means there is only sea in the matrix. 
    The list pair has k operator and each operator has two integer A[i].x, A[i].y 
    means that you can change the grid matrix[A[i].x][A[i].y] from sea 0 to island 1.
    Return how many island are there in the matrix after each operator.You need to return an array of size K.
    
    Input: n = 4, m = 5, A = [[1,1],[0,1],[3,3],[3,4]]
    Output: [1,1,2,2]
     
    @param n: An integer - number of rows
    @param m: An integer - number of cols
    @param operators: an array of Points
    @return: an integer array
    """
    class Point:
        def __init__(self, a=0, b=0):
            self.x = a
            self.y = b

    def num_islands_operators(self, n, m, operators):
        if not operators:
            return []
        done = set()
        ufni2 = UnionFind()
        res = []
        for i, operator in enumerate(operators):
            ufni2.add((operator.x, operator.y))  # can't add operator because: Point(1, 2) == Point(1, 2) is False
            done.add((operator.x, operator.y))
            for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                newx, newy = operator.x + dx, operator.y + dy
                if 0 <= newx < n and 0 <= newy < m and (newx, newy) in done:  # (1, 2) == (1, 2) is True
                    ufni2.union((operator.x, operator.y), (newx, newy))
            res.append(ufni2.count)
        return res

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
    def surrounded_regionns_uf(self, board):
        if not board or not board[0]:
            return
        rows, cols = len(board), len(board[0])
        if rows <= 2 or cols <= 2:
            return
        dummy = (-1, -1)  # index that doesn't exist
        ufo = UnionFind()
        ufo.add(dummy)  # 把dummy point跟所有边线上的O union起来

        for i in range(rows):
            for j in range(cols):
                if board[i][j] == 'O':
                    ufo.add((i, j))
                    if self.inbound(board, (i, j)):
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            newx, newy = i + dx, j + dy
                            if board[newx][newy] == "O":
                                ufo.add((newx, newy))  # it will check if (newx, newy) is already in before adding
                                ufo.union((i, j), (newx, newy))  # will check if they are already connected
                    else:  # on edge
                        ufo.union((i, j), dummy)

        for x in range(rows):
            for y in range(cols):
                if board[x][y] == "O" and not ufo.connected((x, y), dummy):  # 不和dummy相连的island全换成X
                    board[x][y] = "X"

    def inbound(self, matrix, node):
        x, y = node[0], node[1]
        return 0 < x < len(matrix) - 1 and 0 < y < len(matrix[0]) - 1


if __name__ == '__main__':
    uf = UnionFind()
    input_1 = [[1, 1, 0, 0, 0],
               [0, 1, 0, 0, 1],
               [0, 0, 0, 1, 1],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 0, 1]]
    print(uf.num_islands_uf(input_1))  # expect 5
