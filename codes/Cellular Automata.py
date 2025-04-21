import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import gzip

L = 100
sweeps = np.arange((L*100) + 1)
np.random.seed(int(time.time()))

def swap_pos(i,j,k,l):
    state[i,j],state[k,l] = state[k,l],state[i,j]

def scramble(state,number):
    states = []
    ent = []
    for n in range(number):
        ent.append(sys.getsizeof(gzip.compress(state.tobytes())))
        if n in [0,L,L*2,L*4,L*8,L*16,L*32,L*48,L*64,L*80,L*100]:
            states.append(state.copy())
        n_i_rand = np.random.randint(0,L,L**2)
        n_j_rand = np.random.randint(0,L,L**2)
        for y in range(L**2):
            i = n_i_rand[y]
            j = n_j_rand[y]
            for x in range(1000):
                direction = np.random.randint(0,4)
                if direction == 0: # left
                    if j == 0:
                        continue
                    swap_pos(i,j,i,j-1)
                    break
                if direction == 1: # up
                    if i == 0:
                        continue
                    swap_pos(i,j,i-1,j)
                    break
                if direction == 2: # right
                    if j == L-1:
                        continue
                    swap_pos(i,j,i,j+1)
                    break
                if direction == 3: # down
                    if i == L-1:
                        continue
                    swap_pos(i,j,i+1,j)
                    break    
    return(states,ent)

state = np.zeros((L,L))
state[0:L,0:L//2]=1
output,entropy = scramble(state,(L*100) + 1)

entropy_avg = (entropy[L*32] + entropy[L*48] + entropy[L*64] + entropy[L*80])//4

x_markers = [0,L,L*2,L*4,L*8,L*16,L*32,L*48,L*64,L*80]
y_markers = [entropy[0],entropy[L],entropy[L*2],entropy[L*4],entropy[L*8],entropy[L*16],entropy[L*32],entropy[L*48],entropy[L*64],entropy[L*80]]
 
# y_markers = [entropy[0],entropy[L],entropy[L*2],entropy[L*4],entropy[L*8],entropy[L*16],entropy[L*32],entropy[L*48],entropy[L*64],entropy[L*80]]
plt.plot(sweeps, entropy, color="red", alpha=0.5, label = "Entropy (Compressed Size)",zorder=0)
plt.scatter(x_markers, y_markers, color='blue', label = "Markers(scatter)",zorder =1 )
#average line
plt.axhline(y=np.mean(entropy_avg), color='purple', linestyle='-', label = "Mean Entropy",zorder=2)

# Add labels and title
plt.title("Entropy of Diffusion Automata")
plt.xlabel('Sweeps')
plt.ylabel('Compressed Size (Entropy)')

# Display the plot
plt.show()