from heapq import heappush, heappop


class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class SinglyLinkedList:
    """

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
        if not right:  # then i is the length og the original linked list
            # return head
            return head if i < n else head.next

        while right.next:
            left = left.next
            right = right.next

        left.next = left.next.next
        return head

    """
    104
    given k sorted linked lists [2->6->null,5->null,7->null], return 1 sorted list 2->5->6->7->null
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

