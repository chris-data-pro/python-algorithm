# Write a function about permutation


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

        for replace in range(len(nums) - 2, -1, -1):
            if nums[replace + 1] > nums[replace]:
                for using in range(len(nums) - 1, replace, -1):
                    if nums[using] > nums[replace]:
                        nums[replace], nums[using] = nums[using], nums[replace]
                        nums[replace + 1:] = sorted(nums[replace + 1:])
                        break
                break
            else:
                if replace == 0:
                    nums.sort()
        return nums


