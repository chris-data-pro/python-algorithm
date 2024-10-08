import collections


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


class BSTIterator:
    """
    86
    Design an iterator over a bst with the following rules:
    next() returns the next smallest element in the BST. Elements pop in ascending order (i.e. an in-order traversal)

    Input: tree = {10,1,11,#,6,#,12}
    Output: _next().val [1,6,10,11,12]  in-order 从小到大

    """
    def __init__(self, root):
        self.stack = []
        self._to_next_min(root)

    def has_next(self):
        return len(self.stack) > 0

    def _next(self):
        if len(self.stack) == 0:
            return None
        next_node = self.stack.pop()
        self._to_next_min(next_node.right)  # if next_node.right, append all left nodes in to stack, last is smallest
        return next_node

    def _to_next_min(self, root):
        while root:
            self.stack.append(root)
            root = root.left


class SubTree():
    def __init__(self, largest, n, min, max):
        self.largest = largest  # number of nodes in the largest subtree which is a BST
        self.n = n  # total number of nodes under node
        self.min = min  # smallest node value under node
        self.max = max  # biggest node value under node


class BST:
    """
    methods starts with 'bst_' are binary search tree implementations
    others are for normal binary tree
    """

    """
    910
    Given a BT, find the largest subtree which is a BST, largest means subtree with largest number of nodes in it.
    The subtree which you find must be a full binary tree.
    A binary tree is full if each node is either a leaf or possesses exactly two child nodes.
        
    Input: {10,5,15,1,8,#,7}
    Output：3
    
    Explanation:
        10
        / \
       5  15
      / \   \ 
     1   8   7
    The Largest BST Subtree in this case is :
       5
      / \
     1   8. 
    The return value is the subtree's size, which is 3.
    """
    def largest_bst_subtree(self, root):
        res = self.dfs_largest_bst_subtree(root)
        return res.largest

    def dfs_largest_bst_subtree(self, node):
        if not node:
            return SubTree(0, 0, float('inf'), float('-inf'))
        left = self.dfs_largest_bst_subtree(node.left)
        right = self.dfs_largest_bst_subtree(node.right)

        if left.max < node.val < right.min:  # valid BST
            n = left.n + right.n + 1
        else:
            n = float('-inf')
        largest = max(n, left.largest, right.largest)
        return SubTree(largest, n, min(node.val, left.min), max(node.val, right.max))

    """
    bst, insert node of val under root. 
    *** If changing any connections, we need to return the TreeNode ***
    """
    def bst_insert(self, root, key):
        if not root:
            root = TreeNode(key)
        if key < root.val:
            root.left = self.bst_insert(root.left, key)
        if key > root.val:
            root.right = self.bst_insert(root.right, key)
        return root

    """
    1234 - 450
    bst, delete node of val under root
    """
    def bst_delete(self, root, key):
        if not root:
            return
        if root.val == key:
            if (not root.left) and (not root.right): return  # the node is a leaf
            if (not root.left) and root.right: return root.right
            if root.left and (not root.right): return root.left
            rmin = self.bst_get_min(root.right)  # always left get the min
            root.val = rmin.val
            root.right = self.bst_delete(root.right, rmin.val)
        elif root.val > key:  # delete from the left subtree
            root.left = self.bst_delete(root.left, key)
        elif root.val < key:  # delete from the right subtree
            root.right = self.bst_delete(root.right, key)
        return root

    """
    get the the node with min val under node in bst
    """
    def bst_get_min(self, node):
        if not node:
            return
        while node.left:
            node = node.left
        return node

    """
    convert all nodes in a bt into a list, left - root - right
    
    Input: root of tree Node(27)
         27
        /  \
       14   35
      / \   / \ 
    10  19 31  42
    
    Output: [10, 14, 19, 27, 31, 35, 42]  如果是bst，inorder就是从小到大排序
    """
    def inorder_traverse(self, node):
        if not node:
            return []

        res = []
        res += self.inorder_traverse(node.left)
        res.append(node.val)
        res += self.inorder_traverse(node.right)
        return res

    """
    convert all nodes in a bt into a list, root - left - right
    
         27
        /  \
       14   35
      / \   / \ 
    10  19 31  42
    
    [27, 14, 10, 19, 35, 31, 42]
    """
    def preorder_traverse(self, node):
        if not node:
            return []

        res = []
        # res.append(node)  # for further use
        res.append(node.val)
        res += self.preorder_traverse(node.left)
        res += self.preorder_traverse(node.right)
        return res

    """
    453 - 114
    Flatten a binary tree to a fake "linked list" in pre-order traversal.
    use the right pointer in TreeNode as the next pointer in ListNode. and mark the left child of each node to null.
    
         27
        /  \
       14   35
      / \   / \ 
    10  19 31  42
    
    [27, 14, 10, 19, 35, 31, 42]  each node's right points to the next node, left -> null
    """
    def preorder_flatten_right(self, root):
        if not root:
            return
        ans = self.preorder_traverse(root)

        for i in range(1, len(ans)):  # 1 to len(ans) - 1
            ans[i - 1].left = None
            ans[i - 1].right = ans[i]

        return ans[0]

    def preorder_flatten_right_dfs(self, root):
        if not root:
            return
        self.preorder_flatten_right_dfs(root.left)
        self.preorder_flatten_right_dfs(root.right)
        l, r = root.left, root.right
        root.left, root.right = None, l

        p = root
        while p.right:
            p = p.right
        p.right = r

    """
    7
    Serialize and Deserialize BT
    
         27
        /  \ 
       14   35
      / \   / \ 
    10  19 31  42
    
    serialize_preorder: ['27', '14', '10', '#', '#', '19', '#', '#', '35', '31', '#', '#', '42', '#', '#']
    deserialize_preorder: [27, 14, 10, 19, 35, 31, 42]
    
         1
        /  \ 
       2    3
        \ 
         4 
         
    serialize_preorder: ['1', '2', '#', '4', '#', '#', '3', '#', '#']
    deserialize_preorder: [1, 2, 4, 3]
    """
    def serialize_preorder(self, root):
        if not root:
            return ['#']
        ans = []
        ans.append(str(root.val))
        ans += self.serialize_preorder(root.left)
        ans += self.serialize_preorder(root.right)
        return ans

    def deserialize_preorder(self, data):
        ch = data.pop(0)
        if ch == '#':
            return None
        else:
            root = TreeNode(int(ch))
        root.left = self.deserialize_preorder(data)
        root.right = self.deserialize_preorder(data)
        return root

    """
    deserialize inorder array without #
    
    Input: [4, 5, 6, 7]
    
    Output: tree root Node(6)
           6
          / \
         5   7
        /
       4
       
           5
          / \
         4   6
              \
               7
    """
    def array_2_bst(self, data):
        if not data:
            return
        if len(data) == 1:
            return TreeNode(data[0])

        mid = len(data) // 2
        root = TreeNode(data[mid])

        root.left = self.array_2_bst(data[:mid])

        root.right = self.array_2_bst(data[mid+1:])

        return root

    """
         1
        /  \
       2    3
        \
         4 
    
    ['1', '2', '3', '#', '4', '#', '#']
    commented: [['1'], ['2', '3'], ['#', '4', '#', '#']]
    """
    def serialize_horizontal_order(self, root):
        if not root:
            return []
        res = self.horizontal_nodes(root)

        ans = []
        for values in res.values():
            if values.count(None) == len(values) or not values:  # 此层为空 或 全是None
                continue
            for node in values:
                if node:
                    ans.append(str(node.val))
                else:
                    ans.append('#')
        return ans  # ['1', '2', '3', '#', '4', '#', '#']

        # anses = []
        # for items in res.items():
        #     key, values = items[0], items[1]
        #     ans = []
        #     if values.count(None) == len(values) or not values:  # 此层为空 或 全是None
        #         continue
        #     for node in values:
        #         if node:
        #             ans.append(str(node.val))
        #         else:
        #             ans.append('#')
        #     anses.append(ans)
        # return anses  # [['1'], ['2', '3'], ['#', '4', '#', '#']]

    def horizontal_nodes(self, root):
        if not root:
            return []

        res = {0: [root]}  # 层数 -> [本层所有nodes从左到右]
        level = 1
        while res[level - 1]:  # 上一层的 [nodes] 不为空
            if res[level - 1].count(None) == len(res[level - 1]):  # 上一层 [nodes] 都是None
                break
            res[level] = []
            for i in res[level - 1]:
                if not i:  # or not (i.left or i.right):  加这句3下边不会有 '#', '#'
                    continue
                if i.left:
                    res[level].append(i.left)
                else:
                    res[level].append(None)
                if i.right:
                    res[level].append(i.right)
                else:
                    res[level].append(None)
            level += 1
        return res

    """
          1
        /  \
       2    3
        \
         4 
    
    1,2,3,#,4,#,#,#,#,
    """
    def serialize_horizontal_order_dfs(self, root):
        if not root:
            return
        res = {}  # 层数 -> 'node1.val,node2.val,...' 本层所有node.val的str

        def dfs(node, depth):
            if node:
                if depth not in res:
                    res[depth] = ''
                res[depth] += (str(node.val) + ',')
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
            else:
                if depth not in res:
                    res[depth] = ''
                res[depth] += '#,'

        dfs(root, 0)
        return ''.join(res.values())

    """
    check if a binary tree is uni valued
    """
    def is_uni_valued(self, node):
        return len(set(self.inorder_traverse(node))) == 1

    """
    1115
    tree in horizontal order
    
          1
        /  \
       2    3
        \
         4 
    
    [[1], [2, 3], [4]]
    commented: [1, 2.5, 4]
    """
    def horizontal_order(self, node):
        if not node:
            return []

        res = {0: [node]}  # 层数 -> [本层所有nodes]
        level = 1
        while res[level - 1]:
            res[level] = []
            for i in res[level - 1]:
                if i.left:
                    res[level].append(i.left)
                if i.right:
                    res[level].append(i.right)
            level += 1
        print(res)
        return [[x.val for x in y] for y in res.values() if y]
        # return [sum(el.val for el in y) / len(y) for y in res.values() if y]  # Average of Levels in the tree

    """
    1115
    tree in horizontal order dfs
        
          1
        /  \
       2    3
        \
         4 
    
    [[1], [2, 3], [4]]
    commented: [1, 2.5, 4]
    """
    def horizontal_order_dfs(self, root):
        if not root:
            return []
        res = {}  # 层数 -> [本层所有nodes]

        def dfs(node, depth):  # don't need col here, left first below
            if node:
                if depth not in res:
                    res[depth] = []
                res[depth].append(node)
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)

        dfs(root, 0)
        return [[x.val for x in y] for y in res.values() if y]
        # return [sum(el.val for el in y) / len(y) for y in res.values() if y]  # Average of Levels in the tree

    """
    760
    return the values of the nodes you can see from the right side, ordered from top to bottom
    
    Input: {1,2,3,#,5,#,4}  this is the serialize in horizontal order to represent the bt
    Output: [1,3,4] 
    Explanation:
       1            
     /   \
    2     3         
     \     \
      5     4    
    """
    def right_side_view(self, root):
        if not root:
            return []
        res = {}  # 层数 -> 本层最右边的node.val

        def dfs(node, depth):
            if node:
                if depth not in res:
                    res[depth] = None
                if not res[depth]:  # 如果这层还没有->赋予当前node.val  如果这层已经有值->什么都不做
                    res[depth] = node.val
                dfs(node.right, depth + 1)  # right first 先扫右边
                dfs(node.left, depth + 1)

        dfs(root, 0)
        return list(res.values())

    """
    651
    tree in vertical order dfs, for those in the same col, order by depth from top to bottom
    
    Input: {3,9,8,4,0,1,7}
    Output: [[4],[9],[3,0,1],[8],[7]]  
    Explanation:
          3
        /  \
        9   8
      /  \/  \
      4  01   7   If two nodes are in the same row and column, the order should be from left to right.
    """
    def vertical_order_dfs(self, root):
        if not root:
            return []
        res = {}  # 列数 -> [本列所有 (node, 层数) 从上到下]

        def dfs(node, col, depth):
            if node:
                if col not in res:
                    res[col] = []
                res[col].append((node, depth))
                dfs(node.left, col - 1, depth + 1)
                dfs(node.right, col + 1, depth + 1)

        dfs(root, 0, 0)
        return [[x[0].val for x in sorted(res[y], key=lambda x: x[1])] for y in sorted(res)]  # y是col 负-0-正

    """
    all 1 to each node in the bt
    """
    def plus_one(self, node):
        if not node:
            return

        node.val += 1
        self.plus_one(node.left)
        self.plus_one(node.right)

    """
    701
    trim a bst, all nodes val should be between min and max inclusive. result is valid bst
    
    Input: {8,3,10,1,6,#,14,#,#,4,7,13} min=5 max=13
          8
        /   \
       3     10
      /  \     \
      1   6     14
         / \    /
        4   7  13
        
    Output: {8, 6, 10, #, 7, #, 13}
          8
        /   \
       6     10
        \      \
         7      13
    """
    def bst_trim(self, root, minimum, maximum):
        if not root:
            return
        if root.val > maximum:
            return self.bst_trim(root.left, minimum, maximum)
        elif root.val < minimum:
            return self.bst_trim(root.right, minimum, maximum)
        else:
            root.left = self.bst_trim(root.left, minimum, maximum)
            root.right = self.bst_trim(root.right, minimum, maximum)
            return root

    """
    1704
    Given the root of a BST, return the sum of values of all nodes with value between L and R (inclusive).
    
    Input: {10,5,15,3,7,#,18} min=7 max=15
          10
        /   \
       5     15
      /  \     \
      3   7     18
    
    Output: 32 (= 10 + 15 + 7)
    """
    def bst_trim_sum(self, root, L, R):
        if not root:
            return 0
        ans = 0
        if root.val > R:
            ans += self.bst_trim_sum(root.left, L, R)
        elif root.val < L:
            ans += self.bst_trim_sum(root.right, L, R)
        else:
            ans += root.val
            ans += self.bst_trim_sum(root.left, L, R)
            ans += self.bst_trim_sum(root.right, L, R)
        return ans

    """
    1126
    merge 2 bt into a new binary tree. 
    if two nodes overlap, then sum node values up as the new value of the merged node. 
    Otherwise, the NOT null node will be used as the node of new tree.
    
    Inputs:
        Tree 1                     Tree 2                  
              1                         2                             
             / \                       / \
            3   2                     1   3                        
           /                           \   \
          5                             4   7                  

    Output tree:
             3
            / \
           4   5
          / \   \ 
         5   4   7
    """
    def merge_trees(self, t1, t2):
        if not t1:
            return t2
        if not t2:
            return t1

        t = TreeNode(0)
        t.val = (t1.val if t1 else 0) + (t2.val if t2 else 0)
        t.left = self.merge_trees(t1.left, t2.left)
        t.right = self.merge_trees(t1.right, t2.right)

        return t

    """
    175
    invert a bt, left right 对调
    
    Input => Output: 
      1         1
     / \       / \
    2   3  => 3   2
       /       \
      4         4
    """
    def invert_tree(self, node):
        if not node:
            return
        node.left, node.right = node.right, node.left
        self.invert_tree(node.left)
        self.invert_tree(node.right)
        return node

    """
    check 2 trees are symmetric, i.e. a left right mirror
    
    Input: tree1, tree2
      1         1
     / \       / \
    2   3     3   2
       /       \
      4         4
      
    Output: true
    """
    def are_symmetric(self, node1, node2):
        if (not node1) and (not node2):
            return True
        if (not node1) or (not node2):
            return False
        if node1.val != node2.val:
            return False

        return self.are_symmetric(node1.left, node2.right) and self.are_symmetric(node1.right, node2.left)

    """
    468
    check whether a BT is a mirror of itself (i.e., symmetric around its center).
    
    Input:
         1
        / \
       2   2
      / \ / \
      3 4 4 3
      
    Output: true
    """
    def is_symmetric_tree(self, root):
        if not root:
            return True
        return self.are_symmetric(root.left, root.right)

    """
    check 2 trees are the same
    """
    def is_same_tree(self, root1, root2):
        if (not root1) and (not root2):
            return True
        if (not root1) or (not root2):
            return False
        if root1.val != root2.val:
            return False

        return self.is_same_tree(root1.left, root2.left) and self.is_same_tree(root1.right, root2.right)

    """
    get the the node with max val under node in bst
    """
    def bst_get_max(self, node):
        if not node:
            return
        while node.right:
            node = node.right

        return node

    """
    number of nodes under node
    """
    def count_node(self, node):
        if node is None:
            return 0
        return 1 + self.count_node(node.left) + self.count_node(node.right)

    """
    the max depth of the tree
    """
    def max_depth(self, node):
        if node is None:
            return 0
        return 1 + max(self.max_depth(node.left), self.max_depth(node.right))

    """
    95
    check if root is a valid bst, i.e. all left nodes < root < all right nodes
    """
    def bst_is_valid(self, node, minn=None, maxn=None):
        if not node:
            return True
        if minn and node.val <= minn.val:
            return False
        if maxn and node.val >= maxn.val:
            return False

        return self.bst_is_valid(node.left, minn, node) and self.bst_is_valid(node.right, node, maxn)

    """
    find target val in binary search tree
    """
    def bst_val_in_tree(self, node, target):
        if not node:
            return False
        if node.val == target:
            return True  # can return node 返回以target为val的node，和它下面的tree

        if node.val > target:
            return self.bst_val_in_tree(node.left, target)
        if node.val < target:
            return self.bst_val_in_tree(node.right, target)

    """
    find target val in binary tree
    """
    def val_in_tree(self, node, target):
        if not node:
            return False
        if node.val == target:
            return True

        return self.val_in_tree(node.left, target) or self.val_in_tree(node.right, target)

    """
    find the parent node of target
    """
    def find_parent_node(self, node, target):
        if not node:
            return
        if node.left and node.left.val == target:
            return node
        if node.right and node.right.val == target:
            return node

        left = self.find_parent_node(node.left, target)
        right = self.find_parent_node(node.right, target)

        return left if left else right

    """
    bt to graph
    
    Inputs: tree, graph = {}  
          1
        /  \
       2    3
        \
         4 
    
    Output: graph = {1: {2, 3}, 2: {1, 4}, 4: {2}, 3: {1}}  treeNode -> set(treeNode1, treeNode2, ...) all neighbors
    """
    def bt_to_graph_dfs(self, root, graph):  # initialize graph as {} outside this function, {node -> set(neighbor nodes)}
        if not root:
            return
        if root not in graph:
            graph[root] = set()  # no duplicates
        if root.left:
            if root.left not in graph:
                graph[root.left] = set()
            graph[root].add(root.left)
            graph[root.left].add(root)
            self.bt_to_graph_dfs(root.left, graph)
        if root.right:
            if root.right not in graph:
                graph[root.right] = set()
            graph[root].add(root.right)
            graph[root.right].add(root)
            self.bt_to_graph_dfs(root.right, graph)

    """
    given a start node in graph, return all nodes' distances from start
    
    tree
          1
        /  \
       2    3
        \
         4 
    
    Input: graph = {1: {2, 3}, 2: {1, 4}, 4: {2}, 3: {1}}  treeNode -> set(treeNode1, treeNode2, ...) all neighbors
           start node = node(2)

    Output: {start node: 0, neighbor node: 1, neighbor node: 1, neighbor's neighbor: 2, ...}
            {node(2): 0, node(1): 1, node(4): 1, node(3): 2}
    """
    def node_distance_bfs(self, graph, start):  # start is a node
        queue = collections.deque([start])
        distance = {start: 0}  # hash map {node -> node's distance to start int}, no duplicate nodes

        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if neighbor in distance:
                    continue
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)

        return distance

    """
    1506
    given BT, return a list of values of all nodes that distance K from target node
    
    Input: {8,3,10,1,6,#,14,#,#,4,7,13}, target = node(3), K = 2
          8
        /   \
       3     10
      /  \     \
      1   6     14
         / \    /
        4   7  13
        
    Output: [4, 7, 10]
    """
    def distance_k(self, root, target, K):
        graph = {}
        # use dfs to build graph
        self.bt_to_graph_dfs(root, graph)
        distances = self.node_distance_bfs(graph, target)
        return [node.val for node in distances if distances[node] == K]

    """
    find the path to a target in a bst, starting from root
    1. All of the nodes' values will be unique. 2. target will exist in the BST.
    
    Input: tree
         28
        /  \
       20   36
            /
           32
    
    Output: {0: Node(28), 1: Node(36), 2: Node(32)}
    """
    def bst_path_to(self, root, target):  # target是val
        if not root:
            return
        self.path = {}  # 步数 -> Node 从root到target

        def dfs(node, depth):
            if node.val == target:
                self.path[depth] = node
            if node.val > target:
                self.path[depth] = node
                dfs(node.left, depth + 1)
            if node.val < target:
                self.path[depth] = node
                dfs(node.right, depth + 1)

        dfs(root, 0)
        return self.path

    """
    1311
    find the lowest common ancestor (LCA) of two given nodes in the BST. bst的最近公共祖先
    1. All of the nodes' values will be unique. 2. p and q are different and both values will exist in the BST.
    
    Input: {6,2,8,0,4,7,9,#,#,3,5}, node(2), node(4)
          6
        /   \
       2      8
      / \    /  \
     0   4   7   9
        / \
       3   5  
       
    Output: Node(2)
    """
    def bst_lowest_common_ancestor(self, root, p, q):
        path_p = self.bst_path_to(root, p.val)
        path_q = self.bst_path_to(root, q.val)

        for i in range(min(len(path_p), len(path_q))):
            if path_p[i] != path_q[i]:
                i -= 1
                break

        return path_p[i]

    def bst_lowestCommonAncestor(self, root, p, q):
        if root == p or root == q:
            return root
        if p.val < root.val and q.val < root.val:  # p, q 都在左子树
            return self.bst_lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:  # p, q 都在右子树
            return self.bst_lowestCommonAncestor(root.right, p, q)
        return root  # p, q 分别在左右子树，那么root即为结果

    """
    88
    find the lowest common ancestor (LCA) of two given nodes in the BT. bt的最近公共祖先
    
    Input: tree = {4,3,7,#,#,5,6}， A = Node(3)， B = Node(5)
          4
        /   \
       3     7
            /  \
            5   6
    
    Output: Node(4)
    
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        if root is None:
            return None

        if root == A or root == B:
            return root

        left_result = self.lowestCommonAncestor(root.left, A, B)
        right_result = self.lowestCommonAncestor(root.right, A, B)

        # A 和 B 一边一个
        if left_result and right_result:
            return root

        # 左子树有一个点或者左子树有LCA
        if left_result:
            return left_result

        # 右子树有一个点或者右子树有LCA
        if right_result:
            return right_result

        # 左右子树啥都没有
        return None

    """
    94
    max path sum in bt. The path may start and end at any node in the tree.
    
    Input: tree = {10,-5,15,#,#,6,20}
          10
        /   \
       -5    15
            /  \
           6   20
           
    Output: 45
    """
    def max_path_sum(self, root):
        if not root:
            return 0
        self.sums = []  # with self. -> global can be int, str, or any type

        def max_sum(node):
            if not node:
                return 0
            left = max_sum(node.left)
            right = max_sum(node.right)
            node_max = max(node.val, node.val + left, node.val + right, node.val + left + right)
            self.sums.append(node_max)

            return max(node.val, node.val + left, node.val + right)  # 只能加一枝

        max_sum(root)
        return max(self.sums)

    """
    375
    Clone a BT, return a deep copy of it
    """
    def clone_tree(self, root):
        if root is None:
            return None
        clone_root = TreeNode(root.val)  # 这个clone_root 不等于 root
        clone_root.left = self.clone_tree(root.left)
        clone_root.right = self.clone_tree(root.right)
        return clone_root


if __name__ == '__main__':
    bst = BST()
    root_1 = TreeNode(27)
    bst.bst_insert(root_1, 14)
    bst.bst_insert(root_1, 10)
    bst.bst_insert(root_1, 19)
    bst.bst_insert(root_1, 35)
    bst.bst_insert(root_1, 31)
    bst.bst_insert(root_1, 42)
    print(bst.inorder_traverse(root_1))
    serialize_1 = bst.serialize_preorder(root_1)
    print(serialize_1)
    deserialized_1 = bst.deserialize_preorder(serialize_1)
    print(bst.preorder_traverse(deserialized_1))
    bsti = BSTIterator(root_1)
    print(bsti._next().val)
    print(bsti._next().val)
    print(bsti._next().val)
    print()

    bst.plus_one(root_1)
    print(bst.inorder_traverse(root_1))
    node_1 = bst.bst_get_min(root_1)
    print(node_1.val)
    print(bst.count_node(root_1))
    print(bst.max_depth(root_1))
    print('root_1 is bst: {}'.format(bst.bst_is_valid(root_1, None, None)))
    print(bst.horizontal_order(root_1))  # will print intermediate res, expect [[28], [15, 36], [11, 20, 32, 43]]
    deleted_1 = bst.bst_delete(root_1, 11)
    print(bst.horizontal_order_dfs(deleted_1))
    print(bst.vertical_order_dfs(deleted_1))
    parent_of_43 = bst.find_parent_node(root_1, 43)
    if parent_of_43:
        print('The parent of node 43 is {}'.format(parent_of_43.val))
    print()
    trimed = bst.bst_trim(root_1, 16, 40)
    print(bst.horizontal_order_dfs(trimed))  # expect [[28], [20, 36], [32]]
    print(bst.vertical_order_dfs(trimed))
    # bst.bst_path_to returns {0: Node(28), 1: Node(36), 2: Node(32)}
    print('path to 32: {}'.format([(i, bst.bst_path_to(trimed, 32)[i].val) for i in bst.bst_path_to(trimed, 32)]))
    print('path to 32: %s' % [(item[0], item[1].val) for item in bst.bst_path_to(trimed, 32).items()])
    print()

    # root_2
    #                  10
    #                 /  \
    #               -5   15
    #                    / \
    #                   6  20
    root_2 = TreeNode(10)
    root_2.left = TreeNode(-5)
    node_2 = TreeNode(15)
    root_2.right = node_2
    node_2.left = TreeNode(6)
    node_2.right = TreeNode(20)
    print('root_2 is bst: {}'.format(bst.bst_is_valid(root_2, None, None)))
    print(bst.bst_val_in_tree(root_2, 20))
    print(bst.horizontal_order(root_2))  # will print intermediate res, expect [[10], [-5, 15], [6, 20]]
    print(bst.horizontal_order_dfs(root_2))
    print(bst.vertical_order_dfs(root_2))
    print(bst.max_path_sum(root_2))  # expect 45
    print()

    print(bst.is_same_tree(root_1, root_2))
    print()

    root_3 = TreeNode(3)
    node_3 = TreeNode(9)
    node_4 = TreeNode(8)
    root_3.left = node_3
    root_3.right = node_4
    node_3.left, node_3.right = TreeNode(4), TreeNode(0)
    node_3.right.right = TreeNode(2)
    node_4.left, node_4.right = TreeNode(1), TreeNode(7)
    node_4.left.left = TreeNode(5)
    print(bst.horizontal_order(root_3))
    print(bst.serialize_horizontal_order(root_3))
    print((bst.vertical_order_dfs(root_3)))
    print(bst.vertical_order_dfs(root_3))
    print('root_3 is bst: {}'.format(bst.bst_is_valid(root_3)))
    deleted_2 = bst.bst_delete(root_3, 1)
    print(bst.vertical_order_dfs(deleted_2))

    print()

    '''
         1
        /  \
       2    3
        \
         4 
    '''
    root_4 = TreeNode(1)
    root_4.left, root_4.right = TreeNode(2), TreeNode(3)
    root_4.left.right = TreeNode(4)
    serialize_4 = bst.serialize_preorder(root_4)
    print(serialize_4)
    deserialized_4 = bst.deserialize_preorder(serialize_4)
    print(bst.preorder_traverse(deserialized_4))
    print(bst.serialize_horizontal_order(root_4))
    print(bst.horizontal_order(root_4))
    print(bst.serialize_horizontal_order_dfs(root_4))
    print(bst.horizontal_nodes(root_4))

    graph = {}
    # use dfs to build graph
    bst.bt_to_graph_dfs(root_4, graph)
    ans = {}
    for item in graph.items():
        key = item[0].val
        values = set()
        for value in item[1]:
            values.add(value.val)
        ans[key] = values
    print(ans)

    print(bst.node_distance_bfs(graph, root_4.left))

    '''
           6
          / \
         5   7
        /
       4
    '''
    # root_5 = TreeNode(6)
    # root_5.left = TreeNode(5)

    root_5 = bst.array_2_bst([1, 2, 3])
    print(bst.inorder_traverse(root_5))
    print(root_5.val, root_5.left.val, root_5.right.val)





