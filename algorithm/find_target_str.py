# Write a function


class FindTargetStr:
    """
    Given a source str and a target str
    Return the first index of target str, return -1 if not found
    """

    """
    13
    @param source: string
    @param target: string
    @return: return the index or -1
    """
    def first_idx_target_in_source(self, source, target):
        ls, lt = len(source), len(target)

        if ls == lt:
            return 0 if source == target else -1

        for i in range(lt, ls + 1):
            if source[i - lt:i] == target:
                return i - lt

        return -1


if __name__ == '__main__':
    fts = FindTargetStr()
    print(fts.first_idx_target_in_source("abcdabcdefg", "bcd"))
