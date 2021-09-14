# Write a function


class DPWordBreak:
    """
    Given a string s and a dictionary of words dict,
    determine if s can be broken into a sequence of one or more dictionary words.
    """

    """
    107
    @param s: A string
    @param wordSet: A dictionary of words dict
    @return: A boolean
    """
    def word_break(self, s, word_set):
        if not s:
            return True
        if not word_set:
            return False

        n = len(s)
        dp = [False] * (n + 1)
        maxLen = max([len(w) for w in word_set])

        dp[0] = True
        for i in range(1, n + 1):
            for j in range(max(i - maxLen, 0), i):
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[n]


if __name__ == '__main__':
    wb = DPWordBreak()
    print(wb.word_break("helloworld", ["hell", "world", "o"]))
