#


class PrefixSum:
    """
    problems about prefix sum 前缀和
    """

    """
    994
    Given a binary array, like [0,1,1,1,0,1,0]
    find the maximum length of a contiguous subarray with equal number of 0 and 1.
    """
    def max_len_contiguous_subarray_bf(self, nums):
        if not nums or len(nums) < 2:
            return 0

        for k in range(len(nums)):  # k is how many elements to take out
            # fixed-length moving window
            for i in range(k + 1):
                win = nums[i:len(nums) - k + i]
                if 2 * sum(win) == len(win):
                    return len(win)
        return 0

    def max_len_contiguous_subarray(self, nums):
        prefix_sum, longest = 0, 0
        table = {0: 0}  # prefix_sum -> index + 1

        for i, num in enumerate(nums):
            prefix_sum += num if num == 1 else -1  # treat 0 as -1 so that this problem converts to _sum_0

            if prefix_sum in table:
                longest = max(longest, i + 1 - table[prefix_sum])
            else:
                table[prefix_sum] = i + 1

        return longest

    """
    911
    Given an array nums and a target value k, 
    find the maximum length of a subarray that sums to k
    """
    def max_len_contiguous_subarray_sum_k(self, nums, k):
        pre_sum, longest = 0, 0
        table = {0: 0}

        for i, num in enumerate(nums):
            pre_sum += num

            if pre_sum - k in table:
                longest = max(longest, i + 1 - table[pre_sum - k])

            if pre_sum not in table:
                table[pre_sum] = i + 1

        return longest


if __name__ == '__main__':
    ps = PrefixSum()
    print(ps.max_len_contiguous_subarray_bf([0, 1, 1, 1, 0, 1, 0]))
    print(ps.max_len_contiguous_subarray([0, 1, 1, 1, 0, 1, 0]))
    print(ps.max_len_contiguous_subarray_sum_k([1, -1, 5, -2, 3], 3))  # expect 4
