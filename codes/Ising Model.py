import numpy as np
import matplotlib.pyplot as plt
import numba
import stats_ising
import scipy

# L = 81

# grid = np.array([[-1, -1, -1], [1, 1, 1], [1, 1, 1]])

N = 2000

T = 10

Beta = 1/T

num_sweeps = 10000


def Measurer(L, Beta, num_sweeps, N):
    
    # if Beta <= 0.5:
    #     grid = np.random.choice([1,-1], (L,L))
    # else:
    #     grid = np.ones((L,L))
    
    grid = np.ones((L,L))
    
    randflip = np.random.rand(num_sweeps*N) #threshold value
    rand_x = np.random.randint(0,L, num_sweeps*N)
    rand_y = np.random.randint(0,L, num_sweeps*N)
    
    s_list = []
    for counter in range(0,num_sweeps):
        s_list.append(int(sweep_run(grid,N,counter, Beta,rand_x,rand_y,randflip),2))
    # plt.imshow(s_list)
    return s_list
    
@numba.njit
def Energy(spins):
    N = len(spins)
    total_energy = 0  # Initialize a 3x3 grid to store energy values

    # Calculate energy for each cell
    for i in range(N):
        for j in range(N):
            # Get the values of the neighboring cells, wrapping around at the edges
            up = spins[i][j] * spins[(i - 1) % N][j]
            down =  spins[i][j] * spins[(i + 1) % N][j]
            left =  spins[i][j] * spins[i][(j - 1) % N]
            right =  spins[i][j] * spins[i][(j + 1) % N]
            
            # Calculate the energy by summing the neighbors
            total_energy += (up + down + left + right)

    return (total_energy*(-1))/2


@numba.njit
def Magnetization(spins):
    M = 0
    L = len(spins)
    for x in range(L):
        for y in range(L):
            M += spins[x][y]

    return (M/(L**2))**2

def deltaE_inc(spins, spin_to_flip):
    E_old = Energy(spins)
    temp_spins = spins.copy()
    temp_spins[spin_to_flip[0]][spin_to_flip[1]] *= -1
    E_new = Energy(temp_spins)
    return E_new - E_old

@numba.njit
def deltaE(spins, pos):
    L = len(spins)
    (x,y) = pos
    dE = 0
    dE += spins[x][y]*spins[(x + 1 + L)% L][(y + L)% L]
    dE += spins[x][y]*spins[(x - 1 + L)% L][(y + L)% L]
    dE += spins[x][y]*spins[(x + L)% L][(y + 1 + L)% L]
    dE += spins[x][y]*spins[(x + L)% L][(y - 1 + L)% L]
    return 2*dE

@numba.njit
def sweep_run(spins,N,sweeps, Beta, rand_x, rand_y, randflip):
    L = len(spins)
    for i in range(0,N):
        j = i + sweeps*N
        c_prime = deltaE(spins, (rand_x[j],rand_y[j]))
        if np.exp(-1 * Beta * c_prime) > randflip[j]:
            spins[rand_x[j]][rand_y[j]] *= (-1)
    spins_int = ["1" if spins[x][y] == 1 else "0" for x in range(L) for y in range(L)]
    s_int =''.join(spins_int)
    return s_int

def Poss_Energies(L):
    poss_energy_list = []
    
    for state in range(2 ** (L * L)):
        bit_str = bin(state)[2:]
        bit_str = '0' * (L * L - len(bit_str)) + bit_str  # Manually adding zeros
        poss_spins = []
        for i in range(L):
            row = []
            for bit in bit_str[i * L:(i + 1) * L]:
                if bit == '1':
                    row.append(1)
                else:
                    row.append(-1)
            poss_spins.append(row)
        poss_energy_list.append(Energy(np.array(poss_spins)))
    return poss_energy_list

def Poss_Magnetizations(L):
    poss_mag_list = []
    
    for state in range(2 ** (L * L)):
        bit_str = bin(state)[2:]
        bit_str = '0' * (L * L - len(bit_str)) + bit_str  # Manually adding zeros
        poss_spins = []
        for i in range(L):
            row = []
            for bit in bit_str[i * L:(i + 1) * L]:
                if bit == '1':
                    row.append(1)
                else:
                    row.append(-1)
            poss_spins.append(row)
        poss_mag_list.append(Magnetization(np.array(poss_spins)))
    return poss_mag_list

def Poss_Prob(L,poss_energy_list):
    total_prob = 0
    prob_list = []
    for e in poss_energy_list:
        prob = np.exp(-1 * Beta * e)
        prob_list.append(prob)
        total_prob += prob
        
    return prob_list/total_prob

def int_to_spins(num, L):
    bit_int = format(num, "0" + str(L**2) + 'b')
    new_spins = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board
    c = 0
    for bit in bit_int:
        new_spins[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
        c += 1
    
    return new_spins

def spins_to_int(spins, L):
    return int("".join(["1" if spins[x][y] == 1 else "0" for x in range(L) for y in range(L)]),2)

def int_to_energy(int_list,L):
    energy_list = []
    for state in int_list:
        bit_str = bin(state)[2:]
        bit_str = '0' * (L * L - len(bit_str)) + bit_str  # Manually adding zeros
        spins = []
        for i in range(L):
            row = []
            for bit in bit_str[i * L:(i + 1) * L]:
                if bit == '1':
                    row.append(1)
                else:
                    row.append(-1)
            spins.append(row)
        energy_list.append(Energy(np.array(spins)))
    return energy_list

def int_to_mag(int_list,L):
    mag_list = []
    for state in int_list:
        bit_str = bin(state)[2:]
        bit_str = '0' * (L * L - len(bit_str)) + bit_str  # Manually adding zeros
        spins = []
        for i in range(L):
            row = []
            for bit in bit_str[i * L:(i + 1) * L]:
                if bit == '1':
                    row.append(1)
                else:
                    row.append(-1)
            spins.append(row)
        mag_list.append(Magnetization(np.array(spins)))
    return mag_list
  
def spin_coarser(spins, divisor):
    L = len(spins)
    new_grid = np.zeros((L//divisor,L//divisor))
    for i in range(0,L,divisor):
        for j in range(0,L,divisor):
            total = 0
            for i2 in range(divisor):
                for j2 in range(divisor):
                    total += spins[i+i2][j+j2]
            new_grid[i//divisor][j//divisor] = 1 if total > 0 else -1
    return new_grid
 
# Exp_val = (Measurer(L, Beta, num_sweeps, N))

# Theo_val = Poss_Prob(3, Poss_Energies(3))

# WEEK 1 STUFF
# bins = [i for i in range(2**(L**2))]
# plt.hist(Exp_val, bins)
# plt.scatter(bins, Theo_val*(Exp_val.count(2**(L**2)-1)/max(Theo_val)), s=4, c='r') # PC_exp scaled such that its max is the same as the max of PE

# plt.show()


# WEEK 2 STUFF
# HISTOGRAMS

# L = 27

# beta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 17000]

# for i in beta_list:
#     grid_list = Measurer(L, i, num_sweeps, N)
#     energy_list = int_to_energy(grid_list, L)
#     (energy_mean, energy_variance, energy_error, energy_autocorrelation) = stats_ising.Stats(np.array(energy_list))
#     mag_list = int_to_mag(grid_list, L)
#     (mag_mean, mag_variance, mag_error, mag_autocorrelation) = stats_ising.Stats(np.array(mag_list))
    
#     energy_bins = sorted(set(energy_list))
#     mag_bins = sorted(set(mag_list))
    
    
#     plt.title(str(i))
#     plt.hist(energy_list, energy_bins)
#     plt.axvline(energy_mean, c = 'r')
#     print("Beta = " + str(i) + "-->" + str(energy_mean))
#     plt.title("Energy Histogram, Beta = " + str(i))
#     plt.show()

#     plt.title(str(i))
#     plt.hist(mag_list, mag_bins)
#     plt.axvline(mag_mean, c = 'r')
#     print("Beta = " + str(i) + "-->" + str(mag_mean))
#     plt.title("Magnetization Histogram, Beta = " + str(i))
#     plt.show()
    
# <M^2> vs Beta

# mag_mean_list = []
# mag_error_list = []

# for i in beta_list:
#     grid_list = Measurer(L, i, num_sweeps, N)
#     mag_list = int_to_mag(grid_list,L)
#     (mag_mean, mag_variance, mag_error, mag_autocorrelation) = stats_ising.Stats(np.array(mag_list))

#     mag_mean_list.append(mag_mean)
#     mag_error_list.append(mag_error)
    

# plt.errorbar(beta_list,mag_mean_list, yerr=mag_error_list, color='blue', ecolor='red')
# plt.xlabel("Beta")
# plt.ylabel("Average Magnetization")
# plt.title("Average Magnetization vs Beta")
# plt.show()

# Heat Capacity vs Temp

energy_mean_list = []

beta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# for i in beta_list:
#     grid_list = Measurer(L, i, num_sweeps, N)
#     energy_list = int_to_energy(grid_list,L)
#     (energy_mean, energy_variance, energy_error, energy_autocorrelation) = stats_ising.Stats(np.array(energy_list))

#     energy_mean_list.append(energy_mean)

# diff_energy_list = np.gradient(energy_mean_list, beta_list)

# heat_cap_list = (-1 * (np.array(beta_list) ** 2)) * (np.array(diff_energy_list))

# plt.plot(1/np.array(beta_list),heat_cap_list)
# plt.xlabel("Temperature")
# plt.ylabel("Heat Capacity")
# plt.title("Heat Capacity vs Temperature")
# plt.show()


# WEEK 3 STUFF

# L = 81

# beta_list = [0.0, 0.3, 0.4, 0.5, 0.6, 1000]

# grid = np.random.choice([1,-1], (L,L))

# for i in beta_list:
#     grid_list = Measurer(L, i, num_sweeps, N)
#     grid = int_to_spins(grid_list[num_sweeps-1], L)
#     plt.imshow(grid)
#     plt.title("81 x 81, Beta = " + str(i))
#     plt.show()
    
#     coarse_1_grid = spin_coarser(grid,3)
#     plt.imshow(coarse_1_grid)
#     plt.title("27 x 27, Beta = " + str(i))
#     plt.show()
    
#     coarse_2_grid = spin_coarser(coarse_1_grid,3)
#     plt.imshow(coarse_2_grid)
#     plt.title("9 x 9, Beta = " + str(i))
#     plt.show()

# plt.close()

# beta_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

beta_list = np.linspace(0,1,20)

L = 27

mag_mean_list = []

for i in beta_list:
    grid_list = Measurer(L, i, num_sweeps, N)
    mag_list = int_to_mag(grid_list,27)
    (mag_mean, mag_variance, mag_error, mag_autocorrelation) = stats_ising.Stats(np.array(mag_list))

    mag_mean_list.append(mag_mean)

plt.plot(beta_list,mag_mean_list)
plt.xlabel("Beta")
plt.ylabel("Average Magnetization")
plt.title("Average Magnetization vs Beta")

L = 81

coarse_mag_mean_list = []

for i in beta_list:
    grid_list = Measurer(L, i, num_sweeps, N)
    
    new_grid_list = [spins_to_int(spin_coarser(int_to_spins(grid, L),3),27) for grid in grid_list]

    mag_list = int_to_mag(new_grid_list,27)
    (mag_mean, mag_variance, mag_error, mag_autocorrelation) = stats_ising.Stats(np.array(mag_list))

    coarse_mag_mean_list.append(mag_mean)
    
plt.plot(beta_list,coarse_mag_mean_list)
plt.legend(["Native","Coarsened"])
plt.show()

mag_mean_list[0] = 0
mag_mean_list[-1] = 1

mag_function = scipy.interpolate.interp1d(mag_mean_list,np.array(beta_list))

R_beta = scipy.interpolate.interp1d(beta_list,mag_function(coarse_mag_mean_list))

cont_beta_list = np.linspace(0,1,100)

x = np.linspace(0,1,100)

y = x

plt.plot(beta_list, R_beta(beta_list))
plt.plot(x,y)
plt.xlabel("Beta")
plt.ylabel("R(Beta)")
plt.title("Interpolation of R(Beta)")
plt.legend(["R(Beta)","x=y"])


B_iter = 0.5
for i in range(3):
    plt.arrow(B_iter, R_beta(B_iter), R_beta(B_iter)-B_iter,0)
    B_iter = R_beta(B_iter)
    if B_iter < 0 or B_iter > 1:
        break
    plt.arrow(B_iter, B_iter, 0, R_beta(B_iter)-B_iter)

fixed_point = scipy.optimize.root(lambda B: R_beta(B) - B, 0.5).x[0]

R_grad = scipy.interpolate.interp1d(cont_beta_list, np.gradient(R_beta(cont_beta_list), cont_beta_list))

slope = R_grad(fixed_point)

v = np.log(3)/np.log(slope)

print(slope)
print(v)
plt.show()