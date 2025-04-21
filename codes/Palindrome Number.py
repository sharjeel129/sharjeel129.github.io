def isPalindrome(x):

    if x < 0:
        return False
    
    x_rev = str(x)[::-1]
    x_rev_int = int(x_rev)

    if x_rev_int == x:
        return True
    
    else:
        return False