import glob
import os
import subprocess
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt  

N = 25
k = 4

NORTH = 0
EAST = 1    
SOUTH = 2
WEST = 3

num_steps = 10000 # 10000
record_interval = 1000 # 1000
avg_interval = 100 # 100

avalanche_record_start = 3000
avalanche_record_end = 3500 

# Create directories if they don't exist
os.makedirs("sand_states", exist_ok=True)
os.makedirs("sand_state_images", exist_ok=True)

def getNeighbor(x, y, dir):
    xp = x
    yp = y
    if dir == NORTH:
        xp += 0
        yp += -1
    if dir == EAST:
        xp += 1
        yp += 0
    if dir == SOUTH:
        xp += 0
        yp += 1
    if dir == WEST:
        xp += -1
        yp += 0

    return (xp, yp)

state = np.zeros((N,N))

nx_rand=np.random.randint(0,N,num_steps)
ny_rand=np.random.randint(0,N,num_steps)

number_sand = 0
number_avalanche = 0

number_sand_list = []
number_avalanche_list = []

img = []

for step in range(num_steps):

    number_sand = 0
    number_avalanche = 0

    if step % record_interval == 0:
        plt.figure(step//record_interval)
        plt.title("Sand Falling After " + str(step) + " Steps")
        plt.imshow(state)
        np.save("sand_states/sand_states_" + str(step), state)
        plt.savefig("sand_state_images\\sand_image_" + str(step)+".png")

    # Add Sand
    state[nx_rand[step], ny_rand[step]] += 1

    # Deal with overflow
    overflow_list = []
    for i in range(N):
        for j in range(N):
            if state[i,j] >= k:
                overflow_list.append((i,j))

    while len(overflow_list) > 0:
        number_avalanche += 1
        rand_tower = np.random.randint(0, len(overflow_list)) # Choose random overflowing tower

        x,y = overflow_list[rand_tower]
        state[x,y] -= k
        if state[x,y] < k:
            overflow_list.pop(rand_tower)

        for i in range(4):
            xp, yp = getNeighbor(x, y, i)

            if xp >= N or xp < 0 or yp >= N or yp < 0: # Outside of bounds
                # Do nothing
                pass
            else:
                state[xp,yp] += 1
                if state[xp,yp] >= k and overflow_list.count((xp,yp)) == 0:
                    overflow_list.append((xp,yp))
        
    # Calculate number of sand particles
        
    for i in range(N):
        for j in range(N):
            number_sand += state[i,j]
    
    #if step > avalanche_record_start and step < avalanche_record_end:
    #    print(step, ":", number_avalanche)

    number_sand_list.append(number_sand)
    number_avalanche_list.append(number_avalanche)

avg_sand_list = []
avg_avalanche_list = []
max_avalanche_size = max(number_avalanche_list)
avalanche_hist = [0 for i in range(0,max_avalanche_size+1)] # 1 so log plot doesnt break

for i in range(num_steps):
    sand_window_avg = 0
    avalanche_window_avg = 0

    window_start = max(0, i-avg_interval//2)
    window_end = min(num_steps, i+avg_interval//2)
    for j in range(window_start,window_end):
        sand_window_avg += number_sand_list[j]
        avalanche_window_avg += number_avalanche_list[j]
    
    sand_window_avg /= window_end-window_start
    avalanche_window_avg /= window_end-window_start

    avg_sand_list.append(sand_window_avg)
    avg_avalanche_list.append(avalanche_window_avg)


for aval in number_avalanche_list:
    avalanche_hist[aval] += 1

plt.figure(num_steps//record_interval)
plt.xlabel("Steps")
plt.ylabel("Number of Sand Granules (window averaged)")
plt.title("Amount of Sand Over Time")
plt.plot([i for i in range(0,num_steps)], avg_sand_list)  #-
plt.savefig("sand_number.png")

plt.figure(num_steps//record_interval+1)
plt.xlabel("Steps")
plt.ylabel("Avalanche Size (window averaged)")
plt.title("Avalanches Sizes Over Time")
plt.plot([i for i in range(0,num_steps)], avg_avalanche_list) #-
plt.savefig("sand_avalanche.png")

x_list = np.log2([i for i in range(0,max_avalanche_size+1)])
y_list = np.log2(avalanche_hist)

x_list = np.delete(x_list, 0)
y_list = np.delete(y_list, 0)

i = 0
while i < len(y_list):
    #print(i)
    if np.isinf(y_list[i]) or np.isneginf(y_list[i]):
        #print("in")
        x_list = np.delete(x_list, i)
        y_list = np.delete(y_list, i)
        i -= 1
    i += 1

y_unique = []
for y_el in y_list:
    if y_unique.count(y_el) == 0:
        y_unique.append(y_el)

x_unique_avg = []
for y_el in y_unique:
    indices = [i for i, x in enumerate(y_list) if x == y_el]
    x_avg = 0
    for it in indices:
        x_avg += x_list[it]
    x_avg /= len(indices)
    x_unique_avg.append(x_avg)

x_unique_avg = np.array(x_unique_avg)
y_unique = np.array(y_unique)


plt.figure(num_steps//record_interval+2)
plt.xlabel("ln(Number of Avalanches)")
plt.ylabel("ln(Avalanche Size)")
plt.title("Number of Avalanches by Size")
plt.scatter(x_list,y_list)


a, b = np.polyfit(x_unique_avg, y_unique, 1)
print(a,b)
plt.plot(x_unique_avg, a*x_unique_avg+b,c="r")  
plt.text(0.75, 0.75, 'b=' + str(round(a,2)), fontsize=20)

plt.savefig("avalanche_log.png")

plt.show()

N = 200
k = 4

state = np.zeros((N,N))

def getNeighbor(x, y, dir):
    xp = x
    yp = y
    if dir == NORTH:
        xp += 0
        yp += -1
    if dir == EAST:
        xp += 1
        yp += 0
    if dir == SOUTH:
        xp += 0
        yp += 1
    if dir == WEST:
        xp += -1
        yp += 0

    return (xp, yp)

state[N//2, N//2] = 32767

# Deal with overflow

overflow_list = []
for i in range(N):
    for j in range(N):
        if state[i,j] >= k:
            overflow_list.append((i,j))

while len(overflow_list) > 0:
    rand_tower = np.random.randint(0, len(overflow_list)) # Choose random overflowing tower

    x,y = overflow_list[rand_tower]
    state[x,y] -= k
    if state[x,y] < k:
        overflow_list.pop(rand_tower)

    for i in range(4):
        xp, yp = getNeighbor(x, y, i)

        if xp >= N or xp < 0 or yp >= N or yp < 0: # Outside of bounds
            # Do nothing
            pass
        else:
            state[xp,yp] += 1
            if state[xp,yp] >= k and overflow_list.count((xp,yp)) == 0:
                overflow_list.append((xp,yp))

plt.imshow(state)
plt.savefig("large_avalanche_image.png")
plt.show()