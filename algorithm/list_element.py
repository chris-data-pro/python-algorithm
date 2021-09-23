from collections import deque
from itertools import combinations


class ListElement:
    """

    """

    """
    172
    Given an array and a value, 
    remove all occurrences of that value in place and return the new length.
    """
    def remove_element(self, A, elem):
        if not A:
            return 0

        i = 0
        while i < len(A):
            last = len(A) - 1
            if A[i] == elem:
                A[i], A[last] = A[last], A[i]
                A.pop()
            else:
                i += 1

        return len(A)

    """
    135
    Given a set of candidate numbers candidates and a target number target. All numbers are positive integers.
    Find all unique combinations in candidates where the numbers sums to target. (x sum w repeat, x_sum_w_repeat)
    The same repeated number may be chosen from candidates unlimited number of times.
    """
    def combination_sum_repeat_dfs(self, candidates, target):
        results = []
        # 集合为空
        if len(candidates) == 0:
            return results

        # 利用set去重后排序
        candidates = sorted(list(set(candidates)))
        # dfs
        self.dfs_repeat(candidates, target, 0, [], results)
        return results

    def dfs_repeat(self, candidates, target, start, combination,  results):
        # 到达边界
        if target == 0:
            results.append(list(combination))
            return

        # 递归的拆解：挑一个数放入current
        for i in range(start, len(candidates)):
            # 剪枝
            if target < candidates[i]:
                break

            combination.append(candidates[i])
            self.dfs_repeat(candidates, target - candidates[i], i, combination, results)
            combination.pop()

    def combination_sum_repeat_bfs(self, candidates, target):
        candidates.sort()
        results = []
        self.bfs_repeat(candidates, target, results)
        return results

    def bfs_repeat(self, candidates, target, results):
        q = deque([])
        for i in range(len(candidates)):
            # 去重
            if i > 0 and candidates[i] == candidates[i - 1]:
                continue
            q.append([candidates[i]])

        while q:
            tmp = q.popleft()
            if sum(tmp) == target:
                results.append(tmp)
            for i in range(len(candidates)):
                # 去重 && 同时去掉比小于当前遍历的最后一个（也是最大）值，只取大于等于的那些
                if i > 0 and candidates[i] == candidates[i - 1] or candidates[i] < tmp[-1]:
                    continue
                if sum(tmp) + candidates[i] <= target:
                    list_in_q = tmp[:]
                    list_in_q.append(candidates[i])
                    q.append(list_in_q)

    """
    153
    Given an array num and a number target. All numbers are positive integers.
    Find all unique combinations in num where the numbers sum to target. (x sum wo repeat, x_sum_wo_repeat)
    Each number in num can only be used once in one combination
    """
    def combination_sum_wo_repeat(self, num, target):
        results = []
        # 集合为空
        if len(num) == 0:
            return results

        nums = sorted(num)
        self.dfs_wo_repeat(nums, target, 0, [], results)
        return results

    def dfs_wo_repeat(self, candidates, target, start, combination, results):
        # combination sum 例题的出口
        if target == 0:
            results.append(list(combination))
            return

        for i in range(start, len(candidates)):
            if target < candidates[i]:
                break
            # 如果第二次遇到相同的candidates[i] == candidates[i-1]，上一轮从candidates[i-1]开始所有组合都已经试过了
            if i > start and candidates[i] == candidates[i-1]:
                continue
            combination.append(candidates[i])
            self.dfs_wo_repeat(candidates, target - candidates[i], i + 1, combination, results)
            combination.pop()

    """
    82
    Given 2 * n + 1 numbers, every numbers occurs twice except one, return it.
    A = [1,1,2,2,3,4,4] => 3
    A = [0,1,0] => 1
    """
    def single_number_list(self, A):
        l = []
        for a in A:
            if a in l:
                l.remove(a)  # removes the first matching element (which is passed as an argument) from the list.
            else:
                l.append(a)
        return l[0]

    def single_number_dict(self, A):
        d = {}
        for a in A:
            if a in d:
                d.pop(a)  # removes and returns the value of the given key a. (.popitem() removes the last item)
            else:
                d[a] = 1
        return list(d.keys())[0]

    def single_number_set(self, A):
        s = set()
        for num in A:
            if num in s:
                s.remove(num)  # removes the specified element from the set. (.discard(num) no error if not exist)
            else:
                s.add(num)
        return s.pop()

    def single_number_xor(self, A):
        rst = 0
        for i in A:
            rst ^= i  # (1) integer y, 0 ^ y = y (2) same integer x, x ^ x = 0 (3) y ^ x ^ x = x ^ y ^ x = y
        return rst


if __name__ == '__main__':
    le = ListElement()
    print(le.remove_element([0, 4, 4, 0, 0, 2, 4, 4], 4))
    print(le.combination_sum_repeat_bfs([7, 1, 2, 5, 1, 6, 10], 8))
    print(le.combination_sum_repeat_dfs([7, 1, 2, 5, 1, 6, 10], 8))
    print(le.combination_sum_wo_repeat([7, 1, 2, 5, 1, 6, 10], 8))
