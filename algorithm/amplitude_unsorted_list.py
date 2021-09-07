import heapq


class AmplitudeUnsorted:
    """
    amplitude of an array is the difference between the largest and the smallest values it contains.
    """

    """
    amplitude of an array
    """
    def amplitude_unsorted(self, L):
        if len(L) < 2:
            return

        return max(L) - min(L)

    """
    Given a list of seasons ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
    each season is 1/4 part of the L
    Return the top 1 season that have most amplitude.
    @param L: list of integers
    @return: string
    """
    def sort_largest(self, L):  # O(NX), X is the complexity of the mapping
        if not L:
            return
        l = len(L)
        if l < 8:
            return
        seasons = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
        count = l // 4

        idx, amplitude = 0, 0

        for i in range(4):
            a = self.amplitude_unsorted(L[i*count:(i + 1)*count])
            if a > amplitude:
                amplitude = a
                idx = i

        return seasons[idx]

    """
    Given a list of seasons ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
    each season is 1/4 part of the L
    Return the top k seasons that have most amplitude.
    @param L: list of integers
    @param k: integer
    @return: sub list of ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
    """
    def sort_largest_k(self, L, k):  # O(NX + NlogN) = O(4*2count + 4log4)
        if not L:
            return
        l = len(L)
        if l < 8:
            return
        seasons = ["WINTER", "SPRING", "SUMMER", "AUTUMN"]
        count = l // 4

        heap = []
        for i in range(4):
            a = self.amplitude_unsorted(L[i*count:(i + 1)*count])
            if len(heap) < k:
                heapq.heappush(heap, (a, i))
            else:
                heapq.heappushpop(heap, (a, i))  # Push item, then pop and return the smallest item from the heap

        return [seasons[h[1]] for h in heap]  # Result UNSORTED!!!

    """
    1859
    Given an unsorted array A of N integers. 
    In one move, we can choose any element in this array and replace it with any value.
    Return the smallest amplitude of array A that we can achieve by performing at most k moves.
    @param A: a list of integer
    @return: Return the smallest amplitude
    """
    def minimum_amplitude_k_replace(self, A, k):
        if not A:
            return
        start, end = 0, len(A) - 1
        if end <= k:
            return 0
        else:
            end -= k

        A.sort()
        # return min(A[i - 1 - k] - A[i] for i in range(k + 1))
        a = A[end] - A[start]
        i = 0

        while i <= k:
            a = min(a, A[end] - A[start])
            start += 1
            end += 1
            i += 1

        return a


if __name__ == '__main__':
    au = AmplitudeUnsorted()
    print(au.amplitude_unsorted([11, 0, -6, -1, -3, 5]))  # expect 17
    print(au.sort_largest([-1, -10, 10, 5, 30, 15, 20, -10, 30, 10, 29, 20]))  # expect "SUMMER"
    print(au.sort_largest_k([-1, -10, 10, 5, 30, 15, 20, -10, 30, 10, 29, 20], 2))  # expect ["SUMMER" "SPRING"]
    print(au.minimum_amplitude_k_replace([11, 0, -6, -1, -3, 5], 3))  # expect 3
