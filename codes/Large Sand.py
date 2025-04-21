import numpy as np
import matplotlib.pyplot as plt

N = 200
k = 4

state = np.zeros((N,N))

NORTH = 0
EAST = 1    
SOUTH = 2
WEST = 3
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
