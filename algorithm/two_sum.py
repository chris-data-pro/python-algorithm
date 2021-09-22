from itertools import combinations


class TwoSum:
    """
    Given an array of integers,
    find two numbers such that they add up to a specific target number.
    """
    def __init__(self):
        self.sums = {}
        self.nums = []

    def add(self, number):
        for i, num in enumerate(self.nums):
            self.sums[num + number] = (i, len(self.nums))
        self.nums.append(number)

    def find(self, sum):
        return sum in self.sums.keys()

    def find_index(self, sum):
        if self.find(sum):
            return self.sums[sum]
        else:
            return ()

    """
    56
    unsorted array - assume that each input would have exactly one solution
    @param numbers: An unsorted array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2), return [] if not found
    """
    def two_sum_single(self, L, target):  # good for both sorted and unsorted list L
        if not L or target is None:
            return []

        hashmap = {}  # element -> idx
        for ind, num in enumerate(L):  # O(N)
            if target - num in hashmap:
                return [hashmap[target - num], ind]
            hashmap[num] = ind

        return []

    """
    sorted array
    @param numbers: An unsorted array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1, index2] (index1 < index2), return [] if not found
    """
    def two_sum_sorted_list(self, L, target):
        if not L or target is None:
            return []

        for i, l in enumerate(L):
            idx, found = self.binary_search(L[i + 1:], target - l)
            if found == target - l:
                return [i, idx + i + 1]

        return []

    def binary_search(self, nums, target):
        if not nums or target is None:
            return None, None
        start, end = 0, len(nums) - 1

        while start + 1 < end:
            mid = (start + end) // 2

            if nums[mid] < target:
                start = mid
            elif nums[mid] == target:
                end = mid
            else:
                end = mid

        if nums[start] == target:
            return start, target
        if nums[end] == target:
            return end, target
        if target - nums[start] > nums[end] - target:
            return end, nums[end]
        else:
            return start, nums[start]

    """
    1879
    unsorted array - assume that each input could have multiple solutions
    @param numbers: An unsorted array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: list of [index1, index2] (index1 < index2), return [] if not found
    """
    def two_sum_multiple(self, L, target):  # good for both sorted and unsorted list L
        if not L or target is None:
            return []

        hashmap = {}  # element -> idx
        res = []
        for ind, num in enumerate(L):  # O(N)
            if target - num in hashmap:
                res.append([hashmap[target - num], ind])
            hashmap[num] = ind

        return res

    """
    57
    unsorted array - assume there could be multiple solutions
    @param L: an unsorted array of n integer
    @return: list of [x, y, z] unique triplets in the array which gives the sum of target (x + y + z = sum) 
    """
    def three_sum_multiple(self, L, target):  # good for both sorted and unsorted list L
        if not L or target is None:
            return []

        res = []
        for ind, num in enumerate(L):
            rest = L[:ind] + L[ind + 1:]
            pairs = self.two_sum_multiple(rest, target - num)  # list of [idx1, idx2]
            for pair in pairs:
                ans = sorted([num, rest[pair[0]], rest[pair[1]]])
                if ans not in res:
                    res.append(ans)

        return res

    def three_sum_multiple_bf(self, L, target):
        L.sort()
        res = {}
        for c in combinations(L, 3):
            if sum(c) == target and c not in res:
                res[c] = 1
        return [list(x) for x in res.keys()]


class ThreeSum:
    """
    Given an array of integers,
    find all possible 3 numbers such that they add up to a specific target number.
    """
    def __init__(self):
        self.sums = {}  # sum -> list of [x, y, z] (sorted values in nums)
        self.nums = []

    def add(self, number):
        for i, j in combinations(self.nums, 2):
            if i + j + number not in self.sums:
                self.sums[i + j + number] = []
            combination = sorted([i, j, number])
            if combination not in self.sums[i + j + number]:
                self.sums[i + j + number].append(combination)
        self.nums.append(number)

    def find(self, sum):
        return sum in self.sums.keys()

    def find_values(self, sum):
        if self.find(sum):
            return self.sums[sum]
        else:
            return []


if __name__ == '__main__':
    ts = TwoSum()
    ts.add(1)
    ts.add(3)
    ts.add(5)
    print(ts.find(4))  # expect True
    print(ts.find_index(4))  # expect (0, 1)
    print(ts.find(7))  # expect False
    print(ts.find_index(7))  # expect ()
    print(ts.nums)
    print(ts.sums)
    print(ts.two_sum_single([3, 1, 3, 6, 5], 8))  # expect [2, 4]
    print(ts.two_sum_single([-1, 0, 1], 100))  # []
    print(ts.two_sum_sorted_list([-1, 0, 1], 100))  # []
    print(ts.two_sum_multiple([0, -1, 2, -3, 4], 1)) # [[1,2],[3,4]]

    ths = ThreeSum()
    ths.add(2)
    ths.add(7)
    ths.add(11)
    ths.add(15)

    print(ths.find_values(20))
