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
        self.largest = largest
        self.n = n
        self.min = min
        self.max = max


class BST:
    """
    methods starts with 'bst_' are binary search tree implementations
    others are for normal binary tree
    """

    """
    910
    Given a BT, find the largest subtree which is a BST, largest means subtree with largest number of nodes in it.
    """
    def largest_bst_subtree(self, node):
        if not node:
            return SubTree(0, 0, float('inf'), float('-inf'))
        left = self.largest_bst_subtree(node.left)
        right = self.largest_bst_subtree(node.right)

        if left.max < node.val < right.min:  # valid BST
            n = left.n + right.n + 1
        else:
            n = float('-inf')
        largest = max(n, left.largest, right.largest)
        return SubTree(largest, n, min(node.val, left.min), max(node.val, right.max))
        # call self.largest_bst_subtree(root) and return largest

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
            if (not root.left) and (not root.right): return
            if (not root.left) and root.right: return root.right
            if root.left and (not root.right): return root.left
            rmin = self.bst_get_min(root.right)
            root.val = rmin.val
            root.right = self.bst_delete(root.right, rmin.val)
        elif root.val > key:
            root.left = self.bst_delete(root.left, key)
        elif root.val < key:
            root.right = self.bst_delete(root.right, key)

        return root

    """
    convert all nodes in a bt into a list, left - root - right
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
    """
    def preorder_traverse(self, node):
        if not node:
            return []

        res = []
        res.append(node)  # for further use
        res += self.preorder_traverse(node.left)
        res += self.preorder_traverse(node.right)
        return res

    """
    453 - 114
    Flatten a binary tree to a fake "linked list" in pre-order traversal.
    use the right pointer in TreeNode as the next pointer in ListNode. and mark the left child of each node to null.
    """
    def preorder_flatten_right(self, root):
        if not root:
            return
        ans = self.preorder_traverse(root)

        for i in range(1, len(ans)):
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
    """
    def serialize_preorder(self, root):
        if not root:
            return ['#']
        ans = []
        ans.append(str(root.val))
        ans += self.serialize_preorder(root.left)
        ans += self.serialize_preorder(root.right)
        return ans

    def deserialize_preoder(self, data):
        ch = data.pop(0)
        if ch == '#':
            return None
        else:
            root = TreeNode(int(ch))
        root.left = self.deserialize_preoder(data)
        root.right = self.deserialize_preoder(data)
        return root

    """
    check if a binary tree is uni valued
    """
    def is_uni_valued(self, node):
        return len(set(self.inorder_traverse(node))) == 1

    """
    1115
    tree in horizontal order
    """
    def horizontal_order(self, node):
        if not node:
            return []

        res = {0: [node]}
        level = 1
        while res[level - 1]:
            res[level] = []
            for i in res[level - 1]:
                if i.left:
                    res[level].append(i.left)
                if i.right:
                    res[level].append(i.right)
            level += 1

        return [[x.val for x in y] for y in res.values() if y]
        # return [sum(el.val for el in y) / len(y) for y in res.values() if y]  # Average of Levels in the tree

    """
    1115
    tree in horizontal order dfs
    """
    def horizontal_order_dfs(self, root):
        if not root:
            return []
        res = {}

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

    def serialize_horizontal(self, root):
        if not root:
            return
        res = {}

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
    760
    return the values of the nodes you can see from the right side, ordered from top to bottom
    """
    def right_side_view(self, root):
        if not root:
            return []
        res = {}

        def dfs(node, depth):
            if node:
                if depth not in res:
                    res[depth] = None
                if not res[depth]:
                    res[depth] = node.val
                dfs(node.right, depth + 1)  # right first
                dfs(node.left, depth + 1)

        dfs(root, 0)
        return list(res.values())

    """
    651
    tree in vertical order dfs, for those in the same col, order by depth
    """
    def vertical_order_dfs(self, root):
        if not root:
            return []
        res = {}

        def dfs(node, col, depth):
            if node:
                if col not in res:
                    res[col] = []
                res[col].append((node, depth))
                dfs(node.left, col - 1, depth + 1)
                dfs(node.right, col + 1, depth + 1)

        dfs(root, 0, 0)
        return [[x[0].val for x in sorted(res[y], key=lambda x: x[1])] for y in sorted(res)]

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
    invert a bt
    """
    def invert_tree(self, node):
        if not node:
            return
        node.left, node.right = node.right, node.left
        self.invert_tree(node.left)
        self.invert_tree(node.right)
        return node

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
    check 2 trees are symmetric, i.e. a left right mirror
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
    """
    def is_symmetric_tree(self, root):
        if not root:
            return True
        return self.are_symmetric(root.left, root.right)

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
            return True  # can return node

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
    given a start node in BT, return all nodes' distances from start
    """
    def node_distance_bfs(self, graph, start):
        queue = collections.deque([start])
        distance = {start: 0}  # hash map {node -> distance to start int}, no duplicate nodes

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
    """
    def bst_path_to(self, root, target):
        if not root:
            return
        self.path = {}

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
    find the lowest common ancestor (LCA) of two given nodes in the BST.
    1. All of the nodes' values will be unique. 2. p and q are different and both values will exist in the BST.
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
        if p.val < root.val and q.val < root.val:
            return self.bst_lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.bst_lowestCommonAncestor(root.right, p, q)
        return root

    """
    88
    find the lowest common ancestor (LCA) of two given nodes in the BT.
    
    Input: tree = {4,3,7,#,#,5,6}， A = 3， B = 5
    Output: 4
    
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
    max path sum in bt
    The path may start and end at any node in the tree.
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

            return max(node.val, node.val + left, node.val + right)

        max_sum(root)
        return max(self.sums)

    """
    375
    Clone a BT, return a deep copy of it
    """
    def clone_tree(self, root):
        if root is None:
            return None
        clone_root = TreeNode(root.val)
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
    bst.plus_one(root_1)
    print(bst.inorder_traverse(root_1))
    node_1 = bst.bst_get_min(root_1)
    print(node_1.val)
    print(bst.count_node(root_1))
    print(bst.max_depth(root_1))
    print('root_1 is bst: {}'.format(bst.bst_is_valid(root_1, None, None)))
    print(bst.horizontal_order(root_1))
    deleted_1 = bst.bst_delete(root_1, 11)
    print(bst.horizontal_order_dfs(deleted_1))
    print(bst.vertical_order_dfs(deleted_1))
    parent_of_43 = bst.find_parent_node(root_1, 43)
    if parent_of_43:
        print('The parent of node 43 is {}'.format(parent_of_43.val))
    print()
    trimed = bst.bst_trim(root_1, 16, 40)
    print(bst.horizontal_order_dfs(trimed))
    print(bst.vertical_order_dfs(trimed))
    print('path to 32: {}'.format([(i, bst.bst_path_to(trimed, 32)[i].val) for i in bst.bst_path_to(trimed, 32)]))
    print()

    root_2 = TreeNode(10)
    root_2.left = TreeNode(-5)
    node_2 = TreeNode(15)
    root_2.right = node_2
    node_2.left = TreeNode(6)
    node_2.right = TreeNode(20)
    print('root_2 is bst: {}'.format(bst.bst_is_valid(root_2, None, None)))
    print(bst.bst_val_in_tree(root_2, 20))
    print(bst.horizontal_order(root_2))
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
    print((bst.vertical_order_dfs(root_3)))
    print(bst.vertical_order_dfs(root_3))
    print('root_3 is bst: {}'.format(bst.bst_is_valid(root_3)))
    deleted_2 = bst.bst_delete(root_3, 1)
    print(bst.vertical_order_dfs(deleted_2))
    bst.serialize(root_3)

    print()
    bsti = BSTIterator(root_1)
    print(bsti._next().val)



