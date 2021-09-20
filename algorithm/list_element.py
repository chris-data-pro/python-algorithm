# Write


class ListElement:
    """

    """

    """
    172
    Given an array and a value, 
    remove all occurrences of that value in place and return the new length.
    """
    def remove_element(self, A, elem):
        if not A:
            return 0

        i = 0
        while i < len(A):
            last = len(A) - 1
            if A[i] == elem:
                A[i], A[last] = A[last], A[i]
                A.pop()
            else:
                i += 1

        return len(A)


if __name__ == '__main__':
    le = ListElement()
    print(le.remove_element([0, 4, 4, 0, 0, 2, 4, 4], 4))
