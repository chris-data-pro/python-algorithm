from heapq import heappush, heappop


class MinStack:
    """
    12
    Implement a stack with following functions:

    push(val) push val into the stack
    pop() pop the top element and return it
    min() return the smallest number in the stack (will never be called when there is no number in the stack)
    All above should be in O(1) cost
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []  # 单调栈heap,即push时只有元素更小的时候才放入这个栈

    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        self.stack.append(number)
        if not self.min_stack or number <= self.min_stack[0]:
            heappush(self.min_stack, number)

    """
    @return: An integer
    """
    def pop(self):
        mini = self.stack.pop()
        if mini == self.min_stack[0]:  # self.min_stack[0] 为heap的最小值
            heappop(self.min_stack)
        print(self.min_stack)

        return mini

    """
    @return: An integer
    """
    def min(self):
        return self.min_stack[0]


class MaxStack:
    """
    859
    Design a max stack that supports push, pop, top, peekMax and popMax.

    push(x) -- Push element x onto stack.
    pop() -- Remove the element on top of the stack and return it.
    top() -- Get the element on the top.
    peekMax() -- Retrieve the maximum element in the stack.
    popMax() -- Retrieve the maximum element in the stack, and remove it.
    If you find more than one maximum elements, only remove the top-most one.
    The last four operations won't be called when stack is empty.
    """

    def __init__(self):
        self.heap = []
        self.stack = []
        self.popped_set = set()
        self.count = 0

    def push(self, x):
        item = (-x, -self.count)
        self.stack.append(item)
        heappush(self.heap, item)
        self.count += 1

    def _clear_popped_in_stack(self):
        while self.stack and self.stack[-1] in self.popped_set:
            self.popped_set.remove(self.stack[-1])
            self.stack.pop()

    def _clear_popped_in_heap(self):
        while self.heap and self.heap[0] in self.popped_set:
            self.popped_set.remove(self.heap[0])
            heappop(self.heap)

    def pop(self):
        self._clear_popped_in_stack()
        item = self.stack.pop()
        self.popped_set.add(item)
        return -item[0]

    def top(self):
        self._clear_popped_in_stack()
        item = self.stack[-1]
        return -item[0]

    def peekMax(self):
        self._clear_popped_in_heap()
        item = self.heap[0]
        return -item[0]

    def popMax(self):
        self._clear_popped_in_heap()
        item = heappop(self.heap)
        self.popped_set.add(item)
        return -item[0]

