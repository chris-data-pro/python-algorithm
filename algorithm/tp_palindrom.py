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


if __name__ == '__main__':
    p = TPPalindrome()
    print(p.is_palindrome("A man, a plan, a canal: Panama*"))  # expect True
