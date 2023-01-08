import math


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
    def binary_search(self, nums, target):  # O(logN)
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
            return start, target  # the first index of target
        if nums[end] == target:
            return end, target
        if target - nums[start] > nums[end] - target:
            return end, nums[end]
        else:
            return start, nums[start]  # same distance return the smaller

    """
    14
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
    460
    Given a sorted array (ascending order) and a target number,
    find the k closest numbers to target
    closer first, if same distance smaller first
    """
    def k_closest_numbers_1bs(self, A, target, k):  # O(longN + k)
        if not A or not target or k == 0 or k > len(A):
            return []

        idx, closest = self.binary_search(A, target)
        res = [closest]
        left, right = idx - 1, idx + 1

        while len(res) < k:
            if right > len(A) - 1:
                res.append(A[left])
                left -= 1
            elif left < 0:
                res.append(A[right])
                right += 1
            elif abs(target - A[left]) > abs(target - A[right]):
                res.append(A[right])
                right += 1
            else:
                res.append(A[left])
                left -= 1

        return res

    def k_closest_numbers_kbs(self, nums, target, k):  # O(k (longN + N))
        if not nums or target is None or k == 0 or k > len(nums):
            return []

        res = []
        for _ in range(k):
            idx, closest = self.binary_search(nums, target)
            res.append(closest)  # O(1)
            nums.pop(idx)  # O(N)

        return res

    """
    62
    a sorted array is rotated at some pivot, e.g. [4, 5, 6, 7, 0, 1, 2]
    If found the target in the array return its index, otherwise return -1
    no duplicate exists in the array
    """
    def search_rotated_list_wo_duplicates(self, A, target):
        if not A:
            return -1
        left = 0
        right = len(A) - 1
        while left + 1 < right:
            mid = left + (right - left) // 2
            if A[mid] > A[left]:
                # 此时left和mid肯定处在同一个递增数组上
                # 那么就直接运用原始的二分查找
                if A[left] <= target < A[mid]:
                    right = mid
                else:
                    left = mid
            else:
                # 此时mid处于第二个递增数组 left处于第一个递增数组 自然的mid和right肯定处于第二个递增数组上
                # 还是直接运用原始的二分查找思想
                if A[mid] < target <= A[right]:
                    left = mid
                else:
                    right = mid

        if A[left] == target:
            return left
        if A[right] == target:
            return right
        return -1

    """
    63
    a sorted array is rotated at some pivot, e.g. [3, 4, 4, 5, 7, 0, 1, 2]
    If found the target in the array return its index, otherwise return -1
    duplicate is allowed in the array
    """
    def search_rotated_list_w_duplicates(self, A, target):
        if not len(A):
            return False
        # step1: find pivot
        left, right = 0, len(A)-1
        while(left < right):
            mid = left + (right-left)//2
            if A[mid] > A[right]:
                left = mid + 1
            elif A[mid] < A[right]:
                right = mid
            else:
                if (right > 0 and A[right - 1] > A[right]):
                    left = right
                else:
                    right -= 1
        pivot = left
        # step2: split
        if pivot == 0:
            left, right = 0, len(A)-1
        elif target >= A[0]:
            left, right = 0, pivot - 1
        else:
            left, right = pivot, len(A) - 1
        # step3: find target
        while left + 1 < right:
            mid = left + (right - left) // 2
            if A[mid] < target:
                left = mid
            else:
                right = mid
        if A[left] == target:
            return True
        if A[right] == target:
            return True
        return False

    """
    159
    a sorted array is rotated at some pivot, e.g. [4, 5, 6, 7, 0, 1, 2]
    find and return the minimum element
    no duplicate exists in the array
    """
    def min_rotated_list_wo_duplicates(self, nums):
        if not nums:
            return

        left, right = 0, len(nums) - 1
        # 二分法
        while left + 1 < right:
            if nums[left] < nums[right]:
                return nums[left]

            mid = left + (right - left)//2
            # 最小值在[left, mid]
            if nums[left] > nums[mid]:
                right = mid
            # 最小值在(mid, right]
            else:
                left = mid + 1

        return min(nums[left], nums[right])

    """
    160
    a sorted array is rotated at some pivot, e.g. [4, 5, 6, 7, 0, 1, 2]
    find and return the minimum element
    duplicate is allowed in the array
    """
    def min_rotated_list_w_duplicates(self, nums):
        if not nums:
            return

        left, right = 0, len(nums) - 1
        # 二分法
        while left + 1 < right:
            if nums[left] < nums[right]:
                return nums[left]

            mid = left + (right - left)//2
            # 最小值在[left, mid]
            if nums[left] > nums[mid]:
                right = mid
            # 最小值在(mid, right]
            elif nums[left] < nums[mid]:
                left = mid + 1
            # 最小值在[left + 1, right]
            else:
                left += 1
        return min(nums[left], nums[right])
    """
    437
    Given a list of n tasks, i-th task needs tasks[i] hours to finish. There are k persons. 
    Return the shortest time to finish all tasks
    @param tasks: an array of integers
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

    """
    141
    sqrt(int n), return the largest integer x that x*x <= n
    """
    def sqrt_bf(self, n):  # brute force O(N)
        y = 0
        while y * y <= n:
            y += 1

        return y - 1

    def sqrt_bs(self, n):  # O(logN)
        start, end = 0, n

        while start + 1 < end:
            mid = (start + end) // 2

            if mid * mid < n:
                start = mid
            elif mid * mid == n:
                end = mid
            else:
                end = mid

        if end * end <= n:
            return end
        return start

    def sqrt_newton(self, n):  # newton's method O(1)
        """
        求i 使 i * i - b = 0, so f(i) = i * i - b, f_(i) = 2 * i
        i_n+1 = i_n - f(i) / f_(i) = (i_n + b / i_n) / 2
        """
        def f(x):
            return x * x - n

        def f_(x):
            return 2 * x

        last, current = 0, n / 2  # initial guess current such that f(last) is close to 0
        while abs(current - last) > 0.0001:
            last = current
            current = last - f(last) / f_(last)
        return int(current)  # floor


if __name__ == '__main__':
    sl = BSSortedList()
    print(sl.shortest_time_to_finish([3, 2, 4], 2))  # expect 5
    print(sl.binary_search([2, 7, 11, 15], 7))  # expect (1, 7)
    print(sl.binary_search([-1, 0, 1][1:], 1))  # expect (1, 1)
    print(sl.binary_search([], 5))  # expect (None, None)
    print(sl.sqrt_bf(10))
    print(sl.sqrt_bs(10))
    print(sl.sqrt_newton(10))
