

class GreedyAlgorithm:
    """

    """

    """
    116
    given an array of positive integers. initially positioned at the 1st index
    each element represents the max jump length at that position
    Determine if you are able to reach the last index.
    """
    def can_jump_to_end(self, arr):
        far, length = 0, len(arr)
        if length == 0:  # 剪枝提前判断输出
            return False
        elif length == 1:
            return True

        for index in range(length):
            if index <= far:  # 可以跳到index处，则判断并更新最远能到达位置far
                far = max(far, index + arr[index])
            else:
                return False  # 说明无法跳到index位置，肯定不能到最后位置，所以直接False
        return far >= length - 1

    def can_jump_to_end_sbs(self, A):  # step by step
        far, length = 0, len(A)
        if length == 0:
            return False
        elif length == 1:
            return True

        start = end = 0
        while far < length - 1:
            for i in range(start, end + 1):  # 站在起点到终点，逐个计算far
                far = max(far, i + A[i])
            if far <= end:
                return False
            start = end + 1
            end = far
        return far >= length - 1

    """
    117
    given an array of positive integers. initially positioned at the 1st index
    each element represents the max jump length at that position
    return the min number of jumps to reach the last index
    """
    def min_jump_to_end(self, A):
        if not A or len(A) <= 1:
            return 0
        n = len(A)
        start = end = 0
        far = 0
        steps = 0
        while far < n - 1:
            for i in range(start, end + 1):
                far = max(far, i + A[i])
            steps += 1
            if far <= end:
                return -1
            start = end + 1
            end = far

        return steps


if __name__ == '__main__':
    ga = GreedyAlgorithm()
    print(ga.can_jump_to_end([3, 2, 1, 0, 4]))
    print(ga.can_jump_to_end_sbs([3, 2, 1, 0, 4]))
    print(ga.min_jump_to_end([2, 3, 1, 1, 4]))  # expect 2

