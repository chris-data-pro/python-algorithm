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

