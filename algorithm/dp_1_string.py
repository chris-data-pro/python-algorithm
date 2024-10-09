# Write a function


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
def word_break(s, word_set):
    if not s:
        return True
    if not word_set:
        return False

    n = len(s)
    dp = [False] * (n + 1)
    maxLen = max([len(w) for w in word_set])

    dp[0] = True  # dp[x] meaning s[0:x] or the first x characters can be broken into a sequence of dictionary words
    for i in range(1, n + 1):  # to calculate dp[i] one by one
        for j in range(max(i - maxLen, 0), i):  # the range is how many characters we look back, max is maxLen
            if dp[j] and s[j:i] in word_set:  # if the first j letters can, and the s[index j to i - 1] is in set
                dp[i] = True
                break
    print(dp)
    return dp[n]


if __name__ == '__main__':
    # wb = DPWordBreak()
    print(word_break("helloworld", ["hell", "world", "o"]))
