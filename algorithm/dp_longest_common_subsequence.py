# Write a function to calculate change for a given number


class DPLongestCommonSubsequence:
    """
    Given 2 strings
    Return the length of the longest common sub-sequence
    e.g. str1 = "abcde", str2 = "ace" the longest common sub-sequence is "ace", return 3
    """

    """
    @param str1: string
    @param str2: string
    @return: int length of lcs
    """
    def length_of_lcs_bf(self, str1, str2):
        def dp(i, j):
            if i == 0 or j == 0:
                return 0
            if str1[i - 1] == str2[j - 1]:
                return dp(i - 1, j - 1) + 1
            else:
                return max(dp(i, j - 1), dp(i - 1, j))

        return dp(len(str1), len(str2))

    """
    @param str1: string
    @param str2: string
    @return: int length of lcs
    """
    def length_of_lcs(self, str1, str2):
        m, n = len(str1), len(str2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]  # (n + 1) rows X (m + 1) cols

        for i in range(1, n + 1):  # ith row
            for j in range(1, m + 1):  # jth col
                if str2[i - 1] == str1[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])

        return dp[-1][-1]


# if __name__ == '__main__':
#     lcs = LongestCommonSubsequence()
#     print(lcs.length_of_lcs("daabeddbcedeabcbcbec", "daceeaeeaabbabbacedd"))
