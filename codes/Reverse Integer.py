def reverse(x):
    
    if x == 0:
        return 0
    
    if x > 0:
        x_rev = int(str(x)[::-1])
        if x_rev <= (2**31 -1) and x_rev >= (-1*(2**31)):
            return x_rev
        
        else:
            return 0
    
    if x < 0:
        x_pos = -1 * x
        x_pos_rev = int(str(x_pos)[::-1])
        x_rev = -1 * x_pos_rev 
        
        if x_rev <= (2**31 -1) and x_rev >= (-1*(2**31)):
            return x_rev
        
        else:
            return 0
        
        
reverse(-123)