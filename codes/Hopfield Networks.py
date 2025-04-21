import numpy as np
import matplotlib.pyplot as plt
# import numba

# @numba.njit
def Energy(states, biases, weights):
    total_energy = 0.0
    num_neurons = len(states)
    
    for i in range(num_neurons):
        for j in range(num_neurons):
            total_energy += states[i] * weights[i, j] * states[j]
    
    total_energy *= -0.5
    
    for i in range(num_neurons):
        total_energy += states[i] * biases[i]
    
    return total_energy


# @numba.njit
def evolver(states, biases, weights):
    num_neurons = len(states)
    check_interval = num_neurons
    iteration = 0
    energy_list = []
    
    running = True
    while running:
        neuron_idx = np.random.randint(0, num_neurons)
        
        if iteration % check_interval == 0:
            energy_list.append(Energy(states, biases, weights))
        
        states[neuron_idx] = 1 if np.dot(weights[neuron_idx], states) > biases[neuron_idx] else -1
        
        iteration += 1
        running = False
        
        for j in range(num_neurons):
            if (1 if np.dot(weights[j], states) > biases[j] else -1) != states[j]:
                running = True
                break
    
    return iteration, energy_list, states


# @numba.njit
def state_evolver(states, biases, weights):
    num_neurons = len(states)
    running = True
    
    while running:
        neuron_idx = np.random.randint(0, num_neurons)
        states[neuron_idx] = 1 if np.dot(weights[neuron_idx], states) > biases[neuron_idx] else -1
        
        running = False
        for j in range(num_neurons):
            if (1 if np.dot(weights[j], states) > biases[j] else -1) != states[j]:
                running = True
                break
    
    return states

# @numba.njit
def partial_state_evolver(states, biases, weights, fraction=0.1):
    num_neurons = len(states)
    num_updates = max(1, int(fraction * num_neurons))  # Ensure at least one neuron is updated
    
    for _ in range(num_updates):
        neuron_idx = np.random.randint(0, num_neurons)
        states[neuron_idx] = 1 if np.dot(weights[neuron_idx], states) > biases[neuron_idx] else -1
    
    return states

def modify_state(index, states, biases, weights):
    states[index] = 1 if np.dot(weights[index], states) > biases[index] else -1
    return states

def binary_to_state(binary):
    return np.array([1 if bit == "1" else -1 for bit in binary])

def state_to_binary(state):
    return "".join("1" if s == 1 else "0" for s in state)

# @numba.njit
def state_to_board(state):
    size = int(np.sqrt(state.shape[0]))
    return state.reshape((size, size))

# @numba.njit
def board_to_state(board):
    return board.flatten()

# @numba.njit
def compute_weights(images):
    num_images, num_neurons = len(images), len(images[0])
    weights = np.zeros((num_neurons, num_neurons))
    
    for img in images:
        weights += np.outer(img, img)
    
    for i in range(num_neurons):
        weights[i][i] = 0
    
    return weights / num_images


def mask_left_side(state, fraction):
    board = state_to_board(state)
    size = board.shape[0]
    
    for i in range(size):
        board[i, :size // fraction] = -1
    
    return board_to_state(board)


def perturbation(state, num_flips):
    num_neurons = state.shape[0]
    modified_state = state.copy()
    
    indices = np.random.permutation(num_neurons)[:num_flips]
    modified_state[indices] *= -1
    
    return modified_state

def Hamming_Dist(arr1, arr2):
    i = 0
    count = 0
  
    while(i < len(arr1)): 
        if(arr1[i] != arr2[i]): 
            count += 1

        i += 1
    return count

# Energy Measurement

num_neurons = 100
trials = 10

for trial in range(trials):

    states = np.random.choice([-1, 1], size=num_neurons).astype(np.float64)
    biases = 2 * np.random.rand(num_neurons) - 1
    weights = 2 * np.random.rand(num_neurons, num_neurons) - 1

    for row in range(num_neurons):
        for col in range(row, num_neurons):
            if row == col:
                weights[row, col] = 0
            else:
                weights[col, row] = weights[row, col]

    steps, energies, states_final = evolver(states, biases, weights)
    

    plt.plot(range(0, steps, num_neurons), energies)

plt.xlabel("Steps")
plt.ylabel("Energy")
plt.title("Energies over the Evolution of a Hopfield Network")
plt.show()


# Image Processing

# images = ["0000000000000100010000000000000000000000000010000000000000000001110000001000100001000001101000000001",
#           "0001111000000111100000001100000000110000001111111000001100100000110000000011000000001100000000110000"]

# num_neurons = len(images[0])
# image_arrays = [np.array([1 if bit == "1" else -1 for bit in img]) for img in images]
# weights = compute_weights(image_arrays)

# # Apply Left Side Mask
# states = mask_left_side(image_arrays[0], fraction=2).astype(np.float64)
# biases = np.full(num_neurons, 0.5)

# plt.figure(figsize=(15, 5))

# # Initial Masked State
# plt.subplot(1, 3, 1)
# plt.title("Masked State")
# plt.imshow(state_to_board(states))

# # Partial Recovery (Intermediary Step)
# states_partial = partial_state_evolver(states.copy(), biases, weights, 0.2)
# plt.subplot(1, 3, 2)
# plt.title("Intermediate State")
# plt.imshow(state_to_board(states_partial))

# # Full Recovery
# steps, energies, states = evolver(states_partial, biases, weights)
# plt.subplot(1, 3, 3)
# plt.title("Final State")
# plt.imshow(state_to_board(states))

# plt.show()

# # Apply Perturbation
# states = perturbation(image_arrays[0], num_flips=10).astype(np.float64)
# biases = np.full(num_neurons, 0.5)

# plt.figure(figsize=(15, 5))

# plt.subplot(1, 3, 1)
# plt.title("Perturbed State")
# plt.imshow(state_to_board(states))

# # Partial Recovery after Perturbation
# states_partial = partial_state_evolver(states.copy(), biases, weights, 0.2)
# plt.subplot(1, 3, 2)
# plt.title("Intermediate State")
# plt.imshow(state_to_board(states_partial))

# # Full Recovery
# steps, energies, states = evolver(states_partial, biases, weights)
# plt.subplot(1, 3, 3)
# plt.title("Final State after Perturbation")
# plt.imshow(state_to_board(states))

# plt.show()

# Memory Saving

# num_neurons = 100

# k_len = 70
# p_len = 100
# k_list = range(1, k_len)
# p_list = range(1, p_len)

# hamming = np.zeros((k_len - 1, p_len - 1))
# trials = 5

# for p in p_list:
#     for k in k_list:
#         memories = [np.random.choice([1, -1], num_neurons) for j in range(p)]
#         weights = compute_weights(memories)
#         biases = np.ones(num_neurons) * 0.5
        
#         for j in range(trials):
#             rand_memory = memories[np.random.randint(p)]
#             states = perturbation(rand_memory, k).astype(np.float64)
#             states = state_evolver(states, biases, weights).astype(np.float64)
#             hamming[k - 1, p - 1] += Hamming_Dist(states, rand_memory)
        
#         hamming[k - 1, p - 1] /= trials

# plt.matshow(hamming)
# plt.title("Hamming Distance Plot")
# plt.colorbar()
# plt.xlabel("Number of Images")
# plt.ylabel("Number of Corrupted Bits")
# plt.show()

# Graphitz

# n = 6

# memories = [np.random.choice([1,-1], n) for i in range(2)]

# weights = compute_weights(memories)
# biases = np.ones(n)*0

# f = open("graphviz.txt", "w")

# f.write("digraph G {\n")

# for memory in memories:
#     f.write(str(int(state_to_binary(memory),2)) + " [fillcolor=red, style=filled];\n")

# for i in range(2**n):
#     binary = format(i, "0" + str(n) + "b")

#     for j in range(n):
#         new_binary = state_to_binary(modify_state(j, binary_to_state(binary), biases, weights))
#         if binary != new_binary:
#             f.write(str(int(binary,2)) + " -> " + str(int(new_binary,2)) + ";\n")   

# f.write("}")