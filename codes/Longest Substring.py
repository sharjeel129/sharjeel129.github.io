def lengthOfLongestSubstring(s):
    
    i = 0

    counter_list = []
    
    repeater_list = []
    
    counter = 1

    if len(s) == 0:
        return 0
    
    while i < len(s) - 1:
        first = s[i]
        second = s[i + 1]
        
        if first == second:
            # print(f"Found matching pair: {first}{second} at position {i}")
            i += 1  # Skip past this pair and restart
            counter_list.append(counter)
            # print(counter_list)
            repeater_list = []
            counter = 1
            
        elif repeater_list and second in repeater_list:
            # print(f"Found repeated character: {second} at position {i+1}")
            i -= (len(repeater_list)-1)
            counter_list.append(counter)
            # print(counter_list)
            repeater_list = []
            counter = 1
        
        else:
            # print(f"Pair: {first}{second}")
            repeater_list.append(s[i])
            # print(repeater_list)
            counter += 1
            i += 1  # Just move window by one
    
    counter_list.append(counter)
    return max(counter_list)