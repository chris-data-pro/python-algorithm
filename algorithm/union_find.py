# Write a function usinng union find


class UnionFind:
    def __init__(self):
        self.count = 0  # initially 0 disconnected points
        self.parent = {}
        self.weight = {}

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
        ufo.add(dummy)

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
                if board[x][y] == "O" and not ufo.connected((x, y), dummy):
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
