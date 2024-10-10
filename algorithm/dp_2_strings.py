# Write a function to calculate change for a given number


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
def length_of_lcs_bf(str1, str2):
    def dp(i, j):  # meaning the first i letters in str1 and the first j letters in str2, the longest css
        if i == 0 or j == 0:
            return 0
        if str1[i - 1] == str2[j - 1]:  # if the 1th letter in st1 equals the jth letter in str2
            return dp(i - 1, j - 1) + 1  # then the length just increases by 1
        else:
            return max(dp(i, j - 1), dp(i - 1, j))

    return dp(len(str1), len(str2))


"""
@param str1: string
@param str2: string
@return: int length of lcs
"""
def length_of_lcs(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]  # (n + 1) rows X (m + 1) cols
    # ss = ""
    for i in range(1, n + 1):  # ith row
        for j in range(1, m + 1):  # jth col
            if str2[i - 1] == str1[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                # ss += str1[j - 1]
            else:
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])
    # print(ss)
    return dp[-1][-1]


"""
We are given two strings P and Q, each consisting of N lowercase English letters. 
For each position in the strings, we have to choose one letter from either P or Q, 
in order to construct a new string S, such that the number of distinct letters in S is minimal. 

Our task is to find the number of distinct letters in the resulting string S.

For example, if P = "ca" and Q = "ab", S can be equal to: "ca", "cb", "aa" or "ab". 
String "aa" has only one distinct letter ('a'), so the answer is 1 (which is minimal among those strings).

Write a function:
def solution(P, Q)
that, given two strings P and Q, both of length N, returns the minimum number of distinct letters of a string S, 
that can be constructed from P and Q as described above.

Examples:
1. Given P = "abc", Q = "bcd", your function should return 2. 
   All possible strings S that can be constructed are: "abc", "abd", "acc", "acd", "bbc", "bbd", "bcc", "bcd". 
   The minimum number of distinct letters is 2, 
   which be obtained by constructing the following strings: "acc", "bbc", "bbd", "bcc".
   
2. Given P = "axxz", Q = "yzwy", your function should return 2. 
   String S must consist of at least two distinct letters in this case. 
   We can construct S = "yxxy", where the number of distinct letters is equal to 2, 
   and this is the only optimal solution.
   
3. Given P = "bacad", Q = "abada", your function should return 1. 
   We can choose the letter 'a' in each position, so S can be equal to "aaaaa".
   
4. Given P = "amz", Q = "amz", your function should return 3. 
   The input strings are identical, so the only possible S that can be constructed is "amz", 
   and its number of distinct letters is 3.
"""
def solution(P, Q):
    # Implement your solution here
    from functools import lru_cache

    @lru_cache(None)
    def dp(i, used_letters):
        # Base case: if we have processed all characters
        if i == len(P):
            return len(used_letters)

        while P[i] == Q[i] and i <= len(P) - 2:
            used_letters += (P[i],)  # to add tuples
            i += 1

        # Convert used_letters set to tuple for hashing
        used_letters_set = set(used_letters)

        # Case 1: Choose P[i]
        used_letters_p = used_letters_set | {P[i]}  # to add P[i] to set, if it's already in set, do nothing
        result_p = dp(i + 1, tuple(used_letters_p))

        # Case 2: Choose Q[i]
        used_letters_q = used_letters_set | {Q[i]}
        result_q = dp(i + 1, tuple(used_letters_q))

        # Return the minimum of the two choices
        return min(result_p, result_q)

    # Start the recursion from the first position with an empty set of used letters
    return dp(0, tuple())


if __name__ == '__main__':
    print(length_of_lcs("daabeddbcedeabcbcbec", "daceeaeeaabbabbacedd"))  # daced daabed daabbabbce
    print(length_of_lcs("abcdekk", "acexxxk"))  # acek