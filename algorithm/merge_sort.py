# Write a function


class MergeSort:
    """
    Given two sorted integer arrays A and B,
    merge B into A as one sorted array.
    """

    """
    Given two sorted integer arrays A and B,
    merge B into A as one sorted array.
    """
    def merge_2_sorted_lists(self, A, B):
        m, n = len(A), len(B)
        if m == 0 or n == 0:
            return
        i, j = 0, 0
        L = []

        while i < m and j < n:
            if A[i] <= B[j]:
                L.append(A[i])
                i += 1
            else:
                L.append(B[j])
                j += 1

        if i == m and j < n:
            L += B[j:]
        if j == n and i < m:
            L += A[i:]

        # change A in place for LintCode-64
        # for index in range(m + n):
        #     A[index] = L[index]

        return L

    def merge_sort(self, L):
        if not L or len(L) <= 1:
            return L

        mid = len(L) // 2
        left = self.merge_sort(L[:mid])
        right = self.merge_sort(L[mid:])
        return self.merge_2_sorted_lists(left, right)


if __name__ == '__main__':
    ms = MergeSort()
    print(ms.merge_sort([64, 34, 25, 12, 22, 11, 90]))
    Li = [9, 3, 2, 6, 8]
    print(ms.merge_sort(Li))
    print(Li)

    sLi = ms.merge_sort(Li)
    for idx in range(len(Li)):
        Li[idx] = sLi[idx]
    # Li = sLi

    print(Li)
