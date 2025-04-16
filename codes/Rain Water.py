def trap(height):

    n = len(height)
    water = 0

    left_maxes = [0] * n
    left_maxes[0] = height[0]
    
    
    for i in range(1, n):  
        left_maxes[i] = max(left_maxes[i - 1], height[i])  

    print(left_maxes)    

    right_maxes = [0] * n  
    right_maxes[n - 1] = height[n - 1]
    
    
    for i in range(n - 2, -1, -1):
        right_maxes[i] = max(right_maxes[i + 1], height[i])  

    
    print(right_maxes)

    for i in range(n):  
        min_height = min(left_maxes[i], right_maxes[i])  
        
        # print("min_height = " + str(min_height))
        
        trapped = min_height - height[i]
        
        # print("trapped = " + str(trapped))
        
        water += trapped

    return water

print(trap([1,0,2,0,1]))