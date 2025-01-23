class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

"""
Deserialize and construct a binary tree in horizontal order.
Input: A string the level order traversal of each level's nodes' values. (ie, from left to right, level by level)
Output: Contruct the binary tree and return the root node

Example 1:
Input: tree = "{1,2,3}"
Output: TreeNode(1) 
The tree:
    1
   / \
  2   3

Example 2:
Input: tree = "{1,#,2,3} "
Output: TreeNode(1)
The tree:
    1
     \
      2
     /
    3  

@param tree_str: A string with numbers or #s inside a "{}", delimited by comma
@return: The root node of the contructed binary tree
"""
def deserialize_bt(tree_str):
    if not tree_str or tree_str == "{}":
        return None
    
    # Parse input string to get values
    values = tree_str.strip("{}").split(',')
    if not values:
        return None
        
    # Create root node
    root = TreeNode(int(values[0]))
    queue = [root]
    i = 1
    
    # BFS to construct tree
    while queue and i < len(values):
        node = queue.pop(0)
        
        # Add left child
        if i < len(values):
            if values[i] != '#':
                node.left = TreeNode(int(values[i]))
                print(f"{node.val}'s left is {node.left.val}")
                queue.append(node.left)
            else:
                print(f"{node.val}'s left is None")
        i += 1
        
        # Add right child
        if i < len(values):
            if values[i] != '#':
                node.right = TreeNode(int(values[i]))
                print(f"{node.val}'s right is {node.right.val}")
                queue.append(node.right)
            else:
                print(f"{node.val}'s right is None")
        i += 1
            
    return root


"""
Tree in horizontal order

     1
    / \
   2   3
    \
     4

Input: 1 (root node) or tree = {1,2,3,#,4} 
Output: [[1], [2, 3], [4]]

@param root: A Tree
@return: Level order a list of lists of integer
"""
def horizontal_order_bfs(root): # breath first search
    if not root:
        return []
        
    result = {0: [root]}
    level = 1

    while result[level - 1]:
        result[level] = []
        for node in result[level - 1]:
            if node.left:
                # print(f"{node.val}'s left is {node.left.val}")
                result[level].append(node.left)
            if node.right:
                # print(f"{node.val}'s right is {node.right.val}")
                result[level].append(node.right)
        level += 1 

    # result = {0: [root], 1: [TreeNode(2), TreeNode(3)], 2: [TreeNode(4)], 3: []}
    # result.values() = dict_values([ [TreeNode(1)], [TreeNode(2), TreeNode(3)], [TreeNode(4)], [] ])
    return [[x.val for x in nodelist] for nodelist in result.values() if nodelist]


"""
Tree with all leaves (# represents None) in horizontal order

     1
    / \
   2   3
      /
     4

Input: 1 (root node) or tree = {1,2,3,#,#,4} 
Output: [[1], [2, 3], ['#', '#', 4, '#']]

@param root: A Tree
@return: Level order a list of lists of integer
"""
def horizontal_order_all_bfs(root):
    if not root:
        return []
        
    result = {0: [root]}
    level = 1

    while result[level - 1] and result[level - 1].count(None) < len(result[level - 1]):  # 2nd condition means not all None
        result[level] = []
        print(f"\nlevel = {level}")
        for node in result[level - 1]:
            if node is None:
                print(f"None's left is None\nNone's right is None")
                result[level].extend([None, None])
                continue  # skip the rest of the for loop and go to next node in the for loop
            if node.left:
                print(f"{node.val}'s left is {node.left.val}")
                result[level].append(node.left)
            else:
                print(f"{node.val}'s left is None")
                result[level].append(None)
            if node.right:
                print(f"{node.val}'s right is {node.right.val}")
                result[level].append(node.right)
            else:
                print(f"{node.val}'s right is None")
                result[level].append(None)
        level += 1 

    # result = {0: [root], 1: [TreeNode(2), TreeNode(3)], 2: [None, None, TreeNode(4), None], 3: [None, None, None, None, None, None, None, None]}
    # result.values() = dict_values([ [TreeNode(1)], [TreeNode(2), TreeNode(3)], [None, None, TreeNode(4), None], [None, None, None, None, None, None, None, None] ])
    return [[x.val if x else "#" for x in nodelist] for nodelist in result.values() if nodelist.count(None) < len(nodelist)]
    

"""
Calculate number of nodes under the given node

@param root: A TreeNode
@return: Int - the number of nodes under the given node
"""
def bt_node_count(root):
    if not root:
        return 0
    return 1 + bt_node_count(root.left) + bt_node_count(root.right)


if __name__ == '__main__':
    btroot = deserialize_bt("{1,2,3,#,4,#,5,#,#,#,6}")
    """
           1
          / \
         2   3
          \    \
           4    5
                 \
                  6
    """
    print()
    print(f"The number of nodes in this tree: {bt_node_count(btroot)}")  # 6
    print()
    print(horizontal_order_bfs(btroot))  # [[1], [2, 3], [4, 5], [6]]
    print(horizontal_order_all_bfs(btroot))  # [[1], [2, 3], ['#', 4, '#', 5], ['#', '#', '#', '#', '#', '#', '#', 6]]
