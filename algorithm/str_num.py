# About str to num or num to str


class StrNum:
    """
    str -> integer: int, e.g. int('-234') is -234
    integer -> str: str. e.g. str(-234) is '-234'
    """

    """
    188
    Given a number, 
    insert a 5 at any position of the number to make the number largest after insertion.
    """
    def insert_five(self, a):
        num = str(abs(a))
        if a >= 0:
            for i, n in enumerate(num):
                if int(n) < 5:
                    return int(num[:i] + '5' + num[i:])
            return int(num + '5')
        else:
            for i, n in enumerate(num):
                if int(n) > 5:
                    return -int(num[:i] + '5' + num[i:])
            return -int(num + '5')

    """
    408
    Given two binary strings,
    return their sum in  binary notation
    @param a: string - a binary number
    @param b: string - a binary number
    @return: string - the binary result
    """
    def add_binary(self, a, b):
        i, j = len(a) - 1, len(b) - 1

        res = ''
        flag = 0
        while i >= 0 or j >= 0 or flag != 0:
            if i >= 0:
                flag += int(a[i])
                i -= 1
            if j >= 0:
                flag += int(b[j])
                j -= 1
            res = str(flag % 2) + res  # can change to any base <= 10, e.g. 655
            flag = flag // 2

        return res

    def binary_to_decimal_1(self, binary):
        ret = 0
        for char in binary:
            ret = ret * 2 + (ord(char) - ord('0'))  # ord('0') = 48, ord('1') = 49, ... ord('A') = 65, ord('a') = 97
        return ret

    def binary_to_decimal_2(self, binary):
        ans = 0
        for i in range(1, len(binary) + 1):
            ans += int(binary[-i]) * 2 ** (i - 1)
        return ans

    def decimal_to_binary_1(self, decimal):
        res = ''
        while decimal // 2 != 0:
            res = str(decimal % 2) + res
            decimal = decimal // 2
        return str(decimal) + res

    def decimal_to_binary_2(self, decimal):
        from collections import deque
        if not decimal:
            return '0'

        ret = deque([])
        while decimal:
            ret.appendleft(chr(decimal % 2 + ord('0')))
            decimal //= 2
        return ''.join(ret)

    def decimal_to_base_26(self, n):  # n starts from 0
        res = ''
        decimal = n
        if decimal == 0:
            return 'A'

        while decimal:
            if decimal == n:  # write the first char on the right. 0 -> 'A', 1 -> 'B', ... 25 -> 'Z'
                res = chr(decimal % 26 + ord('A')) + res
            else:
                decimal -= 1  # not the first char on the right, need to -1 because we start from 'A'
                res = chr(decimal % 26 + ord('A')) + res
            decimal //= 26
        return res

    """
    1350
    Given a positive integer, return its corresponding column title as appear in an Excel sheet.
    @param n: a integer
    @return: return a string
    """
    def natural_to_excel_col(self, n):  # n starts from 1
        return self.decimal_to_base_26(n - 1)

    """
    655
    Given two non-negative integers represented as strings,
    return their sum as string
    @param a: string - a decimal number
    @param b: string - a decimal number
    @return: string - the decimal result
    """
    def add_decimal(self, a, b):
        i, j = len(a) - 1, len(b) - 1

        res = ''
        flag = 0
        while i >= 0 or j >= 0 or flag != 0:
            if i >= 0:
                flag += int(a[i])
                i -= 1
            if j >= 0:
                flag += int(b[j])
                j -= 1
            res = str(flag % 10) + res  # can change to any base <= 10
            flag = flag // 10

        return res

    """
    419
    Given a roman numeral, convert it to an integer.
    """
    def romen_to_int(self, s):
        mapping = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        res = 0
        for i in range(len(s)):
            if i < (len(s) - 1) and mapping[s[i]] < mapping[s[i + 1]]:
                res -= mapping[s[i]]
            else:
                res += mapping[s[i]]
        return res

    """
    418
    Given an integer, convert it to a roman str
    """
    def int_to_roman(self, n):
        # write your code here
        mapping = {1000: "M", 900: "CM", 500: "D", 400: "CD", 100: "C", 90: "XC", 50: "L", 40: "XL", 10: "X", 9: "IX",
                   5: "V", 4: "IV", 1: "I"}
        result = ""

        for d in mapping:
            result += mapping[d] * (n // d)
            n %= d
        return result

    """
    425
    Given a digit string excluded 0 and 1
    return all possible letter combinations that the number could represent.
    """
    def phone_letter_combinations(self, digits):
        if not digits or digits == '':
            return []

        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}

        # results = ['']
        # for d in digits:
        #     results = [s + c for s in results for c in phone[d]]
        # return results
        # above is same as below

        res = []
        count = 0
        for d in digits:
            if not res:
                for c in phone[d]:
                    res += c
            else:
                for c in phone[d]:
                    for r in res[:count]:
                        res.append(r + c)

            res = res[count:]
            count = len(res)

        return res

    def phone_letter_combinations_dfs(self, digits):
        if not digits or digits == '':
            return []

        phone = {'2': ['a', 'b', 'c'],
                 '3': ['d', 'e', 'f'],
                 '4': ['g', 'h', 'i'],
                 '5': ['j', 'k', 'l'],
                 '6': ['m', 'n', 'o'],
                 '7': ['p', 'q', 'r', 's'],
                 '8': ['t', 'u', 'v'],
                 '9': ['w', 'x', 'y', 'z']}

        res = []
        self.dfs(phone, '', digits, res)
        return res

    def dfs(self, mapping, string, digits, res):
        if len(string) == len(digits):
            res.append(string)
            return
        for letter in mapping[digits[len(string)]]:
            string += letter
            self.dfs(mapping, string, digits, res)
            string = string[:-1]

    """
    414
    Divide two integers without using multiplication, division and mod operator. function of //
    If it will overflow(exceeding 32-bit signed integer representation range), return 2147483647
    """
    def divide(self, dividend, divisor):
        """
        100, 9
        9<<0 = 9
        9<<1 = 18
        9<<2 = 36
        9<<3 = 72
        9<<4 = 144
        这个最大的x是3，所以首先可以找到 1<<3，也就是 8
        剩下的值是 100-(9<<3)=28，找到 18 剩下 10, 再找到 9 剩下 1，结束。
        最终结果 1<<3 + 1<<1 + 1<<0 = 11。
        """
        INT_MAX = (1 << 31) - 1
        INT_MIN = (1 << 31) * -1
        print(INT_MAX, INT_MIN)
        if dividend == 0 and divisor == 0:
            return 0

        if divisor == 0:
            return INT_MAX if dividend > 0 else INT_MIN

        if dividend == 0:
            return 0

        is_negative = (dividend > 0 > divisor) or (dividend < 0 < divisor)
        a = abs(dividend)
        b = abs(divisor)
        result = 0

        while a >= b:
            shift = 0
            while a >= (b << shift):  # b * 2**shift
                shift += 1
            a -= b << shift - 1  # b * 2**(shift - 1)
            result += 1 << shift - 1

        final_result = result if not is_negative else -result
        if final_result > INT_MAX:
            return INT_MAX
        if final_result < INT_MIN:
            return INT_MIN
        return final_result

    """
    420
    The count-and-say sequence is the sequence of integers beginning as follows: 1, 11, 21, 1211, 111221, ...
    1st: 1 is read off as "one 1" or 11.
    2nd: 11 is read off as "two 1s" or 21.
    3rd: 21 is read off as "one 2, then one 1" or 1211.
    Given an integer n, generate the nth sequence.
    """
    def count_and_say(self, n):
        result = '1'
        for _ in range(n - 1):
            result = self.count_and_say_func(result)
        return result

    def count_and_say_func(self, s):
        count = 0
        num = None
        result = ''

        for c in s:
            if num is None:
                count = 1
                num = c
            elif num == c:
                count += 1
            else:
                result += (str(count) + num)
                count = 1
                num = c
        if count != 0:
            result += (str(count) + num)

        return result


if __name__ == '__main__':
    sn = StrNum()
    print(sn.insert_five(-548))  # expect -5458
    print(sn.binary_to_decimal_1('111'))
    print(sn.binary_to_decimal_2('111'))
    print(sn.decimal_to_binary_1(6))
    print(sn.decimal_to_binary_2(6))
    print(sn.add_binary('11', '1'))  # expect '100'
    print(sn.add_decimal('123', '45'))  # expect '168'
    print(sn.add_decimal('19929', '99'))  # expect 20028
    print(sn.decimal_to_base_26(702))
    print(sn.phone_letter_combinations('23'))
    print(sn.count_and_say(5))  # expect '111221'
