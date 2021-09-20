# Write


class Anagram:
    """
    Anagram 相同字母异序词
    Two strings are anagrams if they can be the same after change the order of characters. "ab" <-> "ba"
    or don't change order "ab" <-> "ab"
    """

    """
    171
    Given an array of strings, return all strings that are anagrams.
    Input:["lint", "intl", "inlt", "code"]
    Output:["lint", "inlt", "intl"]
    """
    def anagrams(self, strs):
        memo = {}  # dictionary -> [word1, word2, ...]
        for s in strs:
            dictionary = ''.join(sorted(s))
            if dictionary not in memo:
                memo[dictionary] = [s]
            else:
                memo[dictionary].append(s)

        res = []
        for i in memo:
            if len(memo[i]) > 1:
                res += memo[i]

        return res


if __name__ == '__main__':
    a = Anagram()
    print(a.anagrams(["lint", "intl", "inlt", "code"]))
