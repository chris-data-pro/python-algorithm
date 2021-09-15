from itertools import permutations, combinations


class Permutation:
    """
    permutation implementations
    """

    """
    52
    next permutation in ascending order, when there's no greater order, return the smallest order
    """
    def next_permutation(self, nums):
        if len(nums) <= 1:
            return nums

        for replace in range(len(nums) - 2, -1, -1):  # backward from len(nums) - 2 to 0
            if nums[replace + 1] > nums[replace]:  # if decrease, the smaller one needs to be replaced
                for using in range(len(nums) - 1, replace, -1):  # backward from len(nums) - 1 to target
                    if nums[using] > nums[replace]:  # find the first one from right that's bigger than target
                        nums[replace], nums[using] = nums[using], nums[replace]  # swap
                        nums[replace + 1:] = sorted(nums[replace + 1:])  # sort the right part
                        break  # only need to do it once
                break
            else:
                if replace == 0:
                    nums.sort()
        return nums

    """
    51
    previous permutation in ascending order, when there's no smaller order, return the greatest lexicographic order
    """
    def previous_permutation(self, nums):
        if len(nums) <= 1:
            return nums

        for replace in range(len(nums) - 2, -1, -1):  # backward from len(nums) - 2 to 0
            if nums[replace + 1] < nums[replace]:  # if increase, the bigger one needs to be replaced
                for using in range(len(nums) - 1, replace, -1):  # backward from len(nums) - 1 to target
                    if nums[using] < nums[replace]:  # find the first one from right that's smaller than target
                        nums[replace], nums[using] = nums[using], nums[replace]  # swap
                        nums[replace + 1:] = sorted(nums[replace + 1:], reverse=True)  # reverse sort the right part
                        break  # only need to do it once
                break
            else:
                if replace == 0:
                    nums.sort(reverse=True)
        return nums

    """
    16
    given a list of numbers with duplicates
    find all unique permutations
    @return: list of lists
    """
    def all_unique_permutations(self, nums):
        self.results = []
        self.visited = {i: False for i in range(len(nums))}
        self.dfs([], sorted(nums))
        return self.results

    def dfs(self, path, nums):
        if len(path) == len(nums):
            self.results.append(path[:])
            return

        for i in range(len(nums)):
            if self.visited[i]:
                continue

            if i != 0 and nums[i] == nums[i - 1] and self.visited[i - 1]:
                continue

            self.visited[i] = True
            path.append(nums[i])
            self.dfs(path, nums)
            path.pop()
            self.visited[i] = False


if __name__ == '__main__':
    # unittest.main()
    p = Permutation()
    print(p.all_unique_permutations([5, 1, 4]))  # list of lists
    print(sorted(list(permutations([5, 1, 4]))))  # list of tuples
    print(list(combinations([5, 1, 4], 2)))  # list of tuples
    print(p.next_permutation([5, 1, 4]))  # expect [5, 4, 1]
    print(p.previous_permutation([5, 1, 4]))  # expect [4, 5, 1]
    print(list(reversed([5, 1, 4])))



