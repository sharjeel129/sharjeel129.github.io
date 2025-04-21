import numpy as np
import matplotlib.pyplot as plt

def vasicek(r0, alpha, gamma, beta, T, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    dt = T/float(N)
    rates = [r0]
    for _ in range(N):
        dr = (alpha - gamma*rates[-1]) * dt + np.sqrt(beta) * np.sqrt(dt) * np.random.normal()
        rates.append(rates[-1] + dr)
    return rates


# Parameters
r0 = 0.1  # initial interest rate
alpha = 0.01
gamma = 0.1
beta = 0.0004  # volatility
T = 50 # time horizon
N = 100  # number of steps
dt = T/N
M = 1000 # number of simulations

def B(T):
     return((1/gamma)*(1-np.exp(-gamma*(T))))
 

def A(T):
    return((1/gamma**2)*(B(T) - T)*((alpha*gamma) - (0.5*beta)) - ((beta*(B(T)**2))/(4*gamma)))



# Plotting
plt.figure(figsize=(10, 6))
full = np.zeros((101))
for _ in range(10):
    rates = vasicek(r0, alpha, gamma, beta, T, N)
    full = np.row_stack((full, rates))
    plt.plot(np.linspace(0, T, N+1), rates)

full = np.delete(full, 0, 0)
means = np.mean(full, axis=0)
variance = np.var(full, axis=0)

plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.title('Simulated Interest Rate Paths for Vasicek model')
plt.legend()
plt.grid(True)
plt.show()

mu = np.zeros(N+1)

for t in range(N+1):
    mu[t] = (np.exp(A(t)-r0*(B(t))))
    

plt.plot(np.linspace(0, T, N+1), mu,'b', label = "exact ZCB Price")
plt.plot(np.linspace(0, T, N+1), mu + means - r0 ,'r', label = "ZCB Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Formula vs Monte Carlo ZCB Price for Vasicek model')
plt.legend()
plt.show()