# Template of binary search


class BSSortedList:
    """
    Given a sorted array (ascending order) and a target number,
    search the target in O(logN) time complexity.
    (python sort/sorted is O(NlogN))
    """

    """
    In a sorted list, return the closest element to target
    @param nums: The integer array.
    @param target: Target to find.
    @return: The index of the closest element, the closest element itself   
    """
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
    Given a sorted array (ascending order) and a target number,
    find the first index of this number
    return -1 if not found
    """
    def first_position_of_target(self, nums, target):
        if not nums or target is None:
            return -1

        idx, closest = self.binary_search(nums, target)
        return idx if closest == target else -1

    """
    Given a sorted array (ascending order) and a target number,
    find the k closest numbers to target
    closer first, if same distance smaller first
    """
    def k_closest_numbers(self, nums, target, k):
        if not nums or target is None or k == 0 or k > len(nums):
            return []

        res = []
        for _ in range(k):
            idx, closest = self.binary_search(nums, target)
            res.append(closest)
            nums.pop(idx)

        return res

    """
    Given a list of n tasks, i-th task needs tasks[i] hours to finish. There are k persons. 
    Return the shortest time to finish all tasks
    @param pages: an array of integers
    @param k: A positive integer
    @return: an integer
    """
    def shortest_time_to_finish(self, tasks, k):
        if not tasks:
            return 0

        def can_complete(hours, n, total_hours):  # assume we know the number of total hours allowed
            person = 1
            hour_accumulated = 0

            for hour in hours:
                if hour_accumulated + hour <= total_hours:
                    hour_accumulated += hour
                else:
                    person += 1
                    hour_accumulated = hour

            return person <= n

        start, end = max(tasks), sum(tasks)  # do a BS on a sorted list from start to end

        while start + 1 < end:
            mid = (start + end) // 2

            if can_complete(tasks, k, mid):
                end = mid
            else:
                start = mid

        if can_complete(tasks, k, start):
            return start

        return end


if __name__ == '__main__':
    sl = BSSortedList()
    print(sl.shortest_time_to_finish([3, 2, 4], 2))  # expect 5
    print(sl.binary_search([2, 7, 11, 15], 7))  # expect (1, 7)
    print(sl.binary_search([-1, 0, 1][1:], 1))  # expect (1, 1)
    print(sl.binary_search([], 5))  # expect (None, None)
