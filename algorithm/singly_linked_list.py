from heapq import heappush, heappop


class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class SinglyLinkedList:
    """
    Singly Linked List implementations
    """

    """
    Given one node
    return the length after the node
    """
    def size(self, node):
        cur = node
        cnt = 0
        while cur:
            cnt += 1
            cur = cur.next
        return cnt

    """
    Given one node
    print the linked list after the node
    """
    def printSLL(self, node):
        cur = node
        while cur:
            print(cur.val, end='->')
            cur = cur.next
        print()

    """
    174
    Remove the nth node from the end
    
    Input: list = 1->2->3->4->5->null， n = 2
    Output: 1->2->3->5->null  从后数第二个node是4

    @param head: The first node of linked list.
    @param n: An integer
    @return: The head of linked list.
    """
    def remove_nth_from_end(self, head, n):
        if not head:
            return None
        if n <= 0:
            return head

        # dummy = SinglyLinkedListNode(0)
        # dummy.next = head
        # left, right = dummy, dummy
        left, right = head, head
        i = 0
        while i < n and right:
            right = right.next
            i += 1

        # now i is the total steps taken
        if not right:  # then i is the length of the original linked list
            # return head
            return head if i < n else head.next

        while right.next:
            left = left.next
            right = right.next

        left.next = left.next.next
        return head

    """
    104, 165
    given k sorted linked lists [2->6->null,5->null,7->null], return 1 sorted list 2->5->6->7->null
    
    Input: lists = [2->6->null, 5->null, 7->null]
    Output: 2->5->6->7->null  merge排序
    
    @param lists: a list of head ListNode
    @return: The head of one sorted list.
    """
    def merge_k_sll(self, lists):
        if not lists:
            return None

        heap = []
        for head in lists:
            while head:
                heappush(heap, head.val)  # must push val to sort inside the heap
                head = head.next

        dummy = ListNode(None)
        curr = dummy
        while heap:
            curr.next = ListNode(heappop(heap))
            curr = curr.next
        return dummy.next

    """
    450
    reverse nodes in k-group (k at a time)
    
    Input: list = 1->2->3->4->5->null
    k = 2
    Output: 2->1->4->3->5->null
    
    @param head: a ListNode
    @param k: An integer
    @return: a ListNode
    """
    def reverse_k_group(self, head, k):
        # D->[1->2->3]->[4->5->6]->7 (k = 3)
        # D->[3->2->1]->[6->5->4]->7
        dummy = ListNode(0)
        dummy.next = head  # connect dummy node to head D-> head -> .....

        prev = dummy
        while prev:
            prev = self.reverse_next_k_node(prev, k)

        return dummy.next  # D-> head

    def find_kth_node(self, head, k):
        # head -> n1 -> n2 -> ... ->nk
        curr = head
        count = 0
        for i in range(k):
            if curr is None:
                return curr
            curr = curr.next
        return curr

    def reverse(self, head):
        prev = None
        curr = head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

    def reverse_next_k_node(self, head, k):
        # head -> n1 -> n2 -> ... ->nk -> nk+1
        # head -> nk -> nk-1 -> ... ->n1 -> nk+1
        n1 = head.next
        nk = self.find_kth_node(head, k)
        if nk is None:
            return None
        nk_plus = nk.next
        # Reverse k nodes
        nk.next = None  # separate the nk and nk+1
        nk = self.reverse(n1)  # nk->nk-1->nk-2->......n1
        # Connect head and nk -> nk-1 -> ... ->n1,  n1 and nk+1 -> nk+2 ->...
        head.next = nk
        n1.next = nk_plus

        return n1

    """
    36
    Reverse a linked list from position m to n.  1 ≤ m ≤ n ≤ length_of_list
    
    Input: linked list = 1->2->3->4->5->6->NULL, m = 2, n = 5
    Output: 1->5->4->3->2->6->NULL
    """
    def reverse_between(self, head, m, n):
        if not head:
            return

        if m == 1:  # head变
            start, end = None, None
            curr = head
            for i in range(m, n + 1):
                if i == m:  # 第一轮, 把第一个node标记为end
                    end = curr
                next_node = curr.next
                curr.next = start
                start = curr
                curr = next_node

            end.next = next_node
            return start
        else:  # m > 1 head不变
            prev_node = head
            for _ in range(m - 2):
                prev_node = prev_node.next

            start, end = None, None
            curr = prev_node.next
            if not curr:  # 第m个node已经到达Null，则不需要reverse直接return head
                return head
            for i in range(m, n + 1):
                if i == m:  # 第一轮, 把第一个node标记为end
                    end = curr
                next_node = curr.next
                curr.next = start
                start = curr
                curr = next_node

            prev_node.next = start
            end.next = next_node
            return head

    def dummy_reverse_between(self, head, m, n):
        if not head:
            return

        dummy = ListNode(None)
        dummy.next = head
        prev_node = dummy
        for _ in range(m - 1):
            prev_node = prev_node.next

        start, end = None, None
        curr = prev_node.next
        if not curr:
            return head
        for i in range(m, n + 1):
            if i == m:
                end = curr
            next_node = curr.next
            curr.next = start
            start = curr
            curr = next_node

        prev_node.next = start
        end.next = next_node

        return head if m > 1 else prev_node.next

    """
    105 - 138
    A linked list, each node has an additional random pointer which could point to any node in the list or null.
    Return the head of a DEEP COPY of the original list.
    
    Deep copy should consist of exactly n brand new nodes, 
    where each new node has its value set to the value of its corresponding original node. 
    Both the next and random pointer of the new nodes should point to new nodes in the copied list 
    None of the pointers in the new list should point to nodes in the original list.

    For example, if there are two nodes X and Y in the original list, where X.random --> Y, 
    then for the corresponding two nodes x and y in the copied list, x.random --> y

    @param head: A RandomListNode
    @return: A RandomListNode
    """
    def clone_random_list(self, head):
        if not head:
            return None

        original2copy = {}
        copiedHead = RandomListNode(head.label)
        original2copy[head] = copiedHead

        original = head
        copy = copiedHead

        while original:
            copy.random = original.random
            if original.next:
                copy.next = RandomListNode(original.next.label)
                original2copy[original.next] = copy.next
            else:
                copy.next = None

            original = original.next
            copy = copy.next

        curr = copiedHead
        while curr:
            if curr.random:
                curr.random = original2copy[curr.random]
            curr = curr.next

        return copiedHead


class RandomListNode:
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


if __name__ == '__main__':
    head, n1, n2, n3 = ListNode(0), ListNode(1), ListNode(2), ListNode(3)
    head.next = n1
    n1.next = n2
    n2.next = n3
    sll = SinglyLinkedList()

    sll.printSLL(head)
    print(sll.size(head))  # expect 4
    sll.printSLL(n2)
    print(sll.size(n2))  # expect 2

    sll.printSLL(sll.remove_nth_from_end(head, 4))  # expect 1->2->3->
    sll.printSLL(head)  # expect 0->1->2->3->

    sll.printSLL(sll.remove_nth_from_end(head, 2))  # expect 0->1->3->
    sll.printSLL(head)  # expect 0->1->3->



