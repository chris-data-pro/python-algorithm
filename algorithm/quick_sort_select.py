import heapq


class QuickSortSelect:
    """
    Quick sort an unsorted array
    Find the K-th smallest element in an unsorted array.
    Find the K-th largest element in an unsorted array.
    Find the top K smallest elements in an unsorted array.
    Find the top K largest element in an unsorted array.
    """

    """
    Quick sort an unsorted array.
    @param L: an integer array
    @start: beginning index
    @end: ending index
    @return: nothing
    """
    def quick_sort(self, L, start, end):  # O(NlogN), O(1)
        if start >= end:
            return

        left, right = start, end
        pivot = L[(start + end) // 2]

        while left <= right:
            while left <= right and L[left] < pivot:
                left += 1
            while left <= right and L[right] > pivot:
                right -= 1
            if left <= right:
                L[left], L[right] = L[right], L[left]
                left += 1
                right -= 1

        self.quick_sort(L, start, right)
        self.quick_sort(L, left, end)

    def quick_sort_in_place(self, L):
        self.quick_sort(L, 0, len(L) - 1)

    def quick_sort_return_new(self, L):
        l = L.copy()
        self.quick_sort(l, 0, len(l) - 1)
        return l

    """
    @param k: An integer - the index in the sorted array you want to select
    @param L: An array
    @start: beginning index
    @end: ending index
    @return: after sorted, the index k and the (K+1)th smallest element
    
    Input: [5, -3, 9, 1], 0, 3, 2
    Output: 2, 5
    """
    def quick_select(self, L, start, end, k):  # O(NlogN)  # k is the index
        if start == end:
            return start, L[start]

        left, right = start, end
        pivot = L[(start + end) // 2]

        while left <= right:
            while left <= right and L[left] < pivot:
                left += 1
            while left <= right and L[right] > pivot:
                right -= 1

            if left <= right:
                L[left], L[right] = L[right], L[left]
                left += 1
                right -= 1

        if start <= right and k <= right:
            return self.quick_select(L, start, right, k)
        if left <= end and k >= left:
            return self.quick_select(L, left, end, k)
        return k, L[k]

    def kth_smallest(self, L, k):
        if not L or k <= 0:
            return None
        _, res = self.quick_select(L, 0, len(L) - 1, k - 1)  # quick_select returns the (k - 1 + 1)th smallest element
        return res

    def k_smallest(self, L, k):
        if not L or k <= 0:
            return []
        idx, res = self.quick_select(L, 0, len(L) - 1, k - 1)
        return L[:idx + 1]

    """
    5, 606
    Find K-th largest element in an unsorted array
    Note that it is the kth largest element in the sorted order, not the kth distinct element.
    
    Input:[9,3,2,4,8],3
    Output:4
    Input:[5, -3, 9, 1],2
    Output:5

    @param L: an integer unsorted array
    @param k: an integer from 1 to n
    @return: the kth largest element
    """
    def kth_largest(self, L, k):
        if not L or k <= 0:
            return None
        _, res = self.quick_select(L, 0, len(L) - 1, len(L) - k)  # index len - 1 => 1st largest, len - k => kth largest
        return res

    def k_largest(self, L, k):
        if not L or k <= 0:
            return []
        idx, res = self.quick_select(L, 0, len(L) - 1, len(L) - k)
        return L[idx:]

    """
    @param k: An integer
    @param nums: An array
    @return: the Kth largest element
    """
    def heapq_pop(self, k, nums):  # 79% - O( N + (N-k)logN )
        if not nums or k <= 0:
            return None

        heapq.heapify(nums)  # in-place heapify -> cost O(N) time

        # nums_heap = []
        # for n in nums:  # O(NlogN)
        #     heapq.heappush(nums_heap, n)  # cost O(logN) time

        for _ in range(len(nums) - k):  # O( (N-k)logN )
            heapq.heappop(nums)
        return heapq.heappop(nums)

    def heapq_push_pop(self, k, nums):  # 52%
        if not nums or k <= 0:
            return None

        max_heap = []
        for n in nums:  # O( 2NlogN )
            heapq.heappush(max_heap, n)
            if len(max_heap) > k:
                heapq.heappop(max_heap)

        return max_heap[0]

    def heapq_nlargest(self, k, nums):  # 6%
        if not nums or k <= 0:
            return None

        heapq.heapify(nums)  # in-place heapify -> cost O(N) time
        return heapq.nlargest(k, nums)[-1]

    def python_sort(self, k, nums):  # 79% - O(NlogN)
        if not nums or k <= 0:
            return None

        nums.sort(reverse=True)
        return nums[k - 1]

    def python_sorted(self, k, nums):  # 79% - O(NlogN)
        if not nums or k <= 0:
            return None

        return sorted(nums)[-k]


if __name__ == '__main__':
    qss = QuickSortSelect()
    print(qss.k_smallest([3, 10, 1000, -99, 4, 100], 3))  # expect [-99, 3, 4]
    L = [9, 3, 2, 6, 8]
    print(qss.quick_sort_return_new(L))
    print(L)
    qss.quick_sort_in_place(L)
    print(L)
    print(qss.heapq_pop(3, [9, 3, 2, 6, 8]))
    print(qss.heapq_push_pop(3, [9, 3, 2, 6, 8]))
    print(qss.heapq_nlargest(3, [9, 3, 2, 6, 8]))
    print(qss.python_sort(3, [9, 3, 2, 6, 8]))
    print(qss.python_sorted(3, [9, 3, 2, 6, 8]))
    print(qss.quick_select([5, -3, 9, 1], 0, 3, 2))
