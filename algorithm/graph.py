from collections import deque


class UndirectedGraphNode:
    """
    Each node in the graph contains a label and a list of its neighbors.
    """
    def __init__(self, x):
        self.label = x
        self.neighbors = []


class Graph:
    """
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
    """

    """
    global variables
    """
    def __init__(self):
        self.dict = {}  # val -> node

    """
    137
    Clone an undirected graph. Nodes are labeled uniquely.
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

        root = UndirectedGraphNode(node.label)
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
            mapping[node] = UndirectedGraphNode(node.label)

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


if __name__ == '__main__':
    g = Graph()
