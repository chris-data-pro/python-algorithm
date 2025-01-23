class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
      
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
def horizontal_order_bfs(root):
    if not root:
        return []
        
    result = {0: [root]}
    level = 1

    while result[level - 1]:
        result[level] = []
        for node in result[level - 1]:
            if node.left:
                result[level].append(node.left)
            if node.right:
                result[level].append(node.right)
        level += 1 

    # result = {0: [root], 1: [TreeNode(2), TreeNode(3)], 2: [TreeNode(4)], 3: []}
    # result.values() = dict_values([ [TreeNode(1)], [TreeNode(2), TreeNode(3)], [TreeNode(4)], [] ])
    return [[x.val for x in nodelist] for nodelist in result.values() if nodelist]

if __name__ == '__main__':
    print("")
