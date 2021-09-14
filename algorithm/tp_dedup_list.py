# Write a function


class TPDedupList:
    """
    Dedup a list
    """

    """
    100
    Given a sorted array
    remove the duplicates in place
    @param: nums: An sorted ineger array
    @return: the dedupped array, its length
    """
    def dedup_sorted_list(self, nums):
        if not nums:
            return 0

        slow = 0
        for fast in range(len(nums)):
            if nums[slow] != nums[fast]:
                slow += 1
                nums[slow] = nums[fast]

        return nums[:slow + 1], slow + 1

    """
    521
    Given an unsorted array
    remove the duplicates in place
    @param: nums: An unsorted ineger array
    @return: the dedupped array, its length
    """
    def dedup_unsorted_list(self, nums):
        if not nums:
            return 0

        slow = 0
        memo = {}
        for fast in range(len(nums)):
            if nums[fast] not in memo:  # search in hash map O(1), better than search in list
                memo[nums[fast]] = 1
                nums[slow] = nums[fast]
                slow += 1

        return nums[:slow], slow

    """
    Without in place requirement
    """
    def dedup_list(self, nums):
        unique = set(nums)
        return list(unique), len(unique)


if __name__ == '__main__':
    dl = TPDedupList()
    print(dl.dedup_sorted_list([-15, -7, -6, -1, 1, 2, 6, 11, 15, 15]))  # expect ([-15,-7,-6,-1,1,2,6,11,15], 9)
