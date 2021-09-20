# Write


class TPPalindrome:
    """
    palindrome is a string like "raceecar" or "racecar"
    """

    """
    415
    Given a string, determine if it is a palindrome, 
    considering only alphanumeric 字母和数字的 characters and ignoring cases 忽略大小写.
    Input: "A man, a plan, a canal: Panama"
    Output: true
    Explanation: "amanaplanacanalpanama"
    """
    def is_palindrome(self, s):
        if not s:
            return True
        start, end = 0, len(s) - 1

        while start < end:
            sc, ec = s[start], s[end]  # sc.isalnum() = True only when sc = 'A', 'a' or '1'
            if (97 <= ord(sc) <= 122) or (48 <= ord(sc) <= 57):  # 'a' - 'z' or '0' - '9', do nothing
                sc = sc
            elif 65 <= ord(sc) <= 90:  # 'A' - 'Z', lower case
                sc = sc.lower()
            else:
                start += 1
                continue

            if (97 <= ord(ec) <= 122) or (48 <= ord(ec) <= 57):
                ec = ec
            elif 65 <= ord(ec) <= 90:
                ec = chr(ord(ec) + 32)  # same as lower()
            else:
                end -= 1
                continue

            if sc == ec:
                start += 1
                end -= 1
            else:
                return False

        return True

    """
    200
    Given a string S, 
    find the longest palindromic substring in S
    @param s: input string
    @return: a string as the longest palindromic substring
    """
    def longest_palindrome_bf(self, s):
        ans = ''

        for i in range(len(s) * 2 - 1):
            left = i // 2
            right = left + i % 2
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1
            ans = max(ans, s[left + 1:right], key=len)

        return ans

    def longest_palindrome_manacher(self, s):
        T = '#'.join('^{}$'.format(s))
        pLen = [0] * len(T)
        right, center, maxLen, maxCenter = 0, 0, 0, 0
        for i in range(1, len(T) - 1):
            if i < right:
                pLen[i] = min(pLen[2 * center - i], right - i)
                # if right - i > pLen[2 * center - i] then pLen[i] = pLen[2 * center - i]

            while T[i+(pLen[i]+1)] == T[i-(pLen[i]+1)]:
                pLen[i] += 1

            # update right and center
            if i + pLen[i] > right:
                right, center = i + pLen[i], i

            if pLen[i] > maxLen:
                maxLen = pLen[i]
                maxCenter = i

        print(T)
        print(pLen)
        start = (maxCenter - maxLen) // 2
        return s[start: start + maxLen]


if __name__ == '__main__':
    p = TPPalindrome()
    print(p.is_palindrome("A man, a plan, a canal: Panama*"))  # expect True
    print(p.longest_palindrome_bf("lbcbgdcdcdgdcdcgggggggt"))
    print(p.longest_palindrome_manacher("lbcbgdcdcdgdcdcgggggggt"))
