# Write a function


class FirstMissingPositive:
    """

    """

    """
    189
    Given an unsorted integer array with duplicates
    find the first missing positive integer.
    @param A: An unsorted array of integers with duplicates
    @return: An integer
    """
    def first_missing_positive(self, A):
        B = sorted([x for x in set(A) if x > 0])

        if not B or B[0] > 1:
            return 1

        for i in range(len(B) - 1):
            if B[i + 1] > B[i] + 1:
                return B[i] + 1

        return B[-1] + 1


if __name__ == '__main__':
    fmp = FirstMissingPositive()
    print(fmp.first_missing_positive([-15, -7, -6, -1, 1, 2, 6, 11, 15, 15]))  # expect 3
