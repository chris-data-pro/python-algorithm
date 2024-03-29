from collections import deque, defaultdict


class GraphNode:
    """
    Each node in the graph contains a label and a list of its neighbors.
    """
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class Graph:
    """
    can be directed or undirected
    """

    """
    global variables
    """
    def __init__(self):
        self.dict = {}  # val -> node

    """
    137
    Clone an undirected graph. Nodes are labeled uniquely.
    
    {1,2,4#2,1,4#3,5#4,1,2#5,3} represents follow graph:
    1------2  3
     \     |  |
      \    |  |
       \   |  |
        \  |  |
          4   5
    we use # to split each node information.
    1,2,4 represents that 2, 4 are 1's neighbors
    2,1,4 represents that 1, 4 are 2's neighbors
    3,5 represents that 5 is 3's neighbor
    4,1,2 represents that 1, 2 are 4's neighbors
    5,3 represents that 3 is 5's neighbor
    
    return a deep copied graph, which has the same structure as the original graph, 
    and any changes to the new graph will not have any effect on the original graph.
    
    @param node: A undirected graph node
    @return: A undirected graph node
    """
    def clone_graph(self, node):
        if node is None:
            return None

        if node.label in self.dict:
            return self.dict[node.label]

        root = GraphNode(node.label)
        self.dict[node.label] = root
        for item in node.neighbors:
            root.neighbors.append(self.clone_graph(item))

        return root

    def clone_graph_bfs(self, node):
        root = node
        if node is None:
            return node

        # use bfs algorithm to traverse the graph and get all nodes.
        nodes = self.get_nodes(node)

        # copy nodes, store the old->new mapping information in a hash map
        mapping = {}
        for node in nodes:
            mapping[node] = GraphNode(node.label)

        # copy neighbors(edges)
        for node in nodes:
            new_node = mapping[node]
            for neighbor in node.neighbors:
                new_neighbor = mapping[neighbor]
                new_node.neighbors.append(new_neighbor)

        return mapping[root]

    def get_nodes(self, node):
        q = deque([node])
        result = set([node])
        while q:
            head = q.popleft()
            for neighbor in head.neighbors:
                if neighbor not in result:
                    result.add(neighbor)
                    q.append(neighbor)
        return result

    """
    176
    Given a directed graph, design an algorithm to find out whether there is a route between two nodes s and t
    
    {A,B,D#B,C,D#D,E#C#E}
    A----->B----->C
     \     |
      \    |
       \   |
        \  v
         ->D----->E

    Input:s = B and t = E,
    Output:true
    Input:s = D and t = C,
    Output:false
    
    @param: graph: A list of Directed graph node
    @param: s: the starting Directed graph node
    @param: t: the terminal Directed graph node
    @return: a boolean value
    """

    def directed_has_route(self, graph, s, t):
        countrd = {}
        for x in graph:
            countrd[x] = 0
        return self.dfs_hr(s, countrd, graph, t)

    def dfs_hr(self, i, countrd, graph, t):
        if countrd[i] == 1:
            return False
        if i == t:
            return True
        countrd[i] = 1
        for j in i.neighbors:
            if countrd[j] == 0 and self.dfs_hr(j, countrd, graph, t):
                return True
        return False

    """
    616
    There are a total of n courses you have to take, labeled from 0 to n - 1.
    prerequisites：to take course 1 you have to first take course 0, which is expressed as a pair: [1,0]
    Given the total number of courses numCourses， and a list of prerequisite pairs, 
    return the ordering of courses you should take to finish all courses.

    There may be multiple correct orders, you just need to return one of them. 
    If it is impossible to finish all courses, return an empty array.
    
    Input: n = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]] 有方向directed graph
    0 --> 1 --> 3
    |           ^
    --> 2 ______|
    
    Output: [0,1,2,3] or [0,2,1,3]

    @param: numCourses: a total of n courses  - integer
    @param: prerequisites: a list of prerequisite pairs - list of lists
    @return: the course order - list of numbers
    """
    def directed_find_order(self, numCourses, prerequisites):
        graph = [[] for i in range(numCourses)]
        in_degree = [0] * numCourses

        # 建图 directed graph
        for node_in, node_out in prerequisites:
            graph[node_out].append(node_in)
            in_degree[node_in] += 1

        num_choose = 0
        queue = deque()
        topo_order = []

        # 将入度为 0 的编号加入队列
        for i in range(numCourses):
            if in_degree[i] == 0:
                queue.append(i)

        while queue:
            now_pos = queue.popleft()
            topo_order.append(now_pos)
            num_choose += 1
            # 将每条邻边删去，如果入度降为 0，再加入队列
            for next_pos in graph[now_pos]:
                in_degree[next_pos] -= 1
                if in_degree[next_pos] == 0:
                    queue.append(next_pos)

        if num_choose == numCourses:
            return topo_order
        return []

    """
    178
    Given n nodes labeled from 0 to n - 1 and a list of undirected edges 
    write a function to check whether these edges make up a valid tree.
    You can assume that no duplicate edges will appear in edges. 
    Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.
    
    Inputs: n = 5， edges = [[0, 1], [0, 2], [0, 3], [1, 4]]  无方向undirected graph
       0 - 1 - 4
      / \
      2  3
    Output: True
    
    Inputs: n = 5， edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]
       0 - 1 - 4
          / \
         2 - 3
    Output: False
    
    n个点，m条边。空间复杂度为O(n^2)。
    建图时每条边都会访问 1 次，搜索时每个点都会被询问1次，时间复杂度为O(max(n, m))。
    """
    def valid_tree(self, n, edges):  # 树是连通的，没有环的
        if len(edges) != n - 1:  # 已知给定的边不重复，n个结点的树一定有(n - 1)个边，所以如果 len(edges) > (n - 1) 一定有环
            return False

        neighbors = defaultdict(list)  # x -> [x所有的neighbors]
        for u, v in edges:
            neighbors[u].append(v)
            neighbors[v].append(u)  # {0: [1], 1: [0, 2, 3, 4], 2: [1, 3], 3: [2, 1], 4: [1]} 第二个例子的edges

        visited = {}
        queue = deque()

        queue.append(0)
        visited[0] = True
        while queue:
            cur = queue.popleft()
            visited[cur] = True
            for node in neighbors[cur]:
                if node not in visited:
                    visited[node] = True
                    queue.append(node)

        return len(visited) == n  # 如果 len(visited) 不等于 n 说明不联通


if __name__ == '__main__':
    g = Graph()

    # {0,1,2,3,4#1,3,4#2,1,4#3,4#4}, 4, 1
    node0 = GraphNode(0)
    node1 = GraphNode(1)
    node2 = GraphNode(2)
    node3 = GraphNode(3)
    node4 = GraphNode(4)

    node0.neighbors = [node1, node2, node3, node4]
    node1.neighbors = [node3, node4]
    node2.neighbors = [node1, node4]
    node3.neighbors = [node4]

    print(g.directed_has_route([node0, node1, node2, node3, node4], node4, node1))  # expect False
