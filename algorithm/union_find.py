# Write a function usinng union find


class UnionFind:
    def __init__(self):
        self.count = 0  # initially 0 disconnected points
        self.parent = {}
        self.weight = {}

    def add(self, item):
        if item in self.parent:
            return
        self.parent[item] = item  # initially each point's parent is itself
        self.weight[item] = 1  # initially each tree has 1 point
        self.count += 1

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
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


if __name__ == '__main__':
    uf = UnionFind()
    input_1 = [[1, 1, 0, 0, 0],
               [0, 1, 0, 0, 1],
               [0, 0, 0, 1, 1],
               [1, 0, 1, 0, 0],
               [1, 0, 0, 0, 1]]
    print(uf.num_islands(input_1))  # expect 5
