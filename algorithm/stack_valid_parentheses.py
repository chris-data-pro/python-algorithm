#


class StackValidParentheses:
    """
    stack: Last-In/First-Out
    """

    """
    423
    Given a string containing just the characters '(', ')', '{', '}', '[' and ']', 
    determine if the input string is valid.
    @param s: A string
    @return: boolean - whether the string is a valid parentheses
    """
    def is_valid_parentheses(self, ps):
        openx = '{[('
        mapping = {')': '(', ']': '[', '}': '{'}
        stack = []

        for p in ps:
            if p in openx:
                stack.append(p)
            elif not stack or stack.pop() is not mapping[p]:
                return False

        return not stack

    """
    193
    Given a string containing just the characters '(' and ')', 
    find the length of the longest valid (well-formed) parentheses substring.
    """
    def longest_valid_parentheses(self, s):
        stack = [-1]
        maxlen = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    maxlen = max(maxlen,  i - stack[-1])
        return maxlen


if __name__ == '__main__':
    svp = StackValidParentheses()
    print(svp.is_valid_parentheses('([)]'))  # expect False
    print(svp.is_valid_parentheses('{[()}'))  # expect False
    print(svp.is_valid_parentheses('{[()]}'))  # expect True
    print(svp.longest_valid_parentheses(")((()))(()"))
