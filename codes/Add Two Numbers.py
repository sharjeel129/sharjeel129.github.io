def addTwoNumbers(l1, l2):
        head = current = ListNode(0)
        carryover = 0

        while any([l1, l2, carryover]):
            sum_digits = carryover
            if l1:
                sum_digits += l1.val
                l1 = l1.next
            if l2:
                sum_digits += l2.val
                l2 = l2.next

            carryover, digit = divmod(sum_digits, 10)
            current.next = ListNode(digit)
            current = current.next

        return head.next