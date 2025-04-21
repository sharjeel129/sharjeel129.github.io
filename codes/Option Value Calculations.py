import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Define parameters
S = 100  # spot price
K = 95   # strike price
T_minus_t = 3 / 12  # time to expiry in years
r = 0.05  # risk-free interest rate
sigma = 0.2  # volatility
NTS = range(1, 51)  # number of time steps

# Define delta t
delta_t = T_minus_t / max(NTS)

# Calculate u, v, and p using the given parameterization
u = 1 + sigma * np.sqrt(delta_t)
v = 1 - sigma * np.sqrt(delta_t)
p = 0.5

# Define function to calculate option value using the binomial model
def binomial_option_value(S, K, u, v, p, r, delta_t, NTS):
    option_values = []
    for N in NTS:
        # Initialize arrays for stock prices and option values
        stock_prices = np.zeros(N + 1)
        option_prices = np.zeros(N + 1)

        # Calculate stock prices at each node
        for i in range(N + 1):
            stock_prices[i] = S * (u ** (N - i)) * (v ** i)

        # Calculate option values at expiry
        for i in range(N + 1):
            option_prices[i] = max(0, stock_prices[i] - K)

        # Backward induction to calculate option values at previous time steps
        for j in range(N - 1, -1, -1):
            for i in range(j + 1):
                option_prices[i] = np.exp(-r * delta_t) * (p * option_prices[i] + (1 - p) * option_prices[i + 1])

        option_values.append(option_prices[0])

    return option_values

# Calculate option values using the binomial model
option_values = binomial_option_value(S, K, u, v, p, r, delta_t, NTS)

# Plot option values vs. number of time steps
plt.plot(NTS, option_values)
plt.xlabel('Number of Time Steps')
plt.ylabel('Option Value')
plt.title('Option Value vs. Number of Time Steps')
plt.grid(True)
plt.show()

S = 100                             # current stock price (spot price)
K = 100                             # exercise price (strike price)
T = 3/12                            # expiry time in years
r = 0.05                            # annualized risk-free interest rate 

sigma = np.arange(0.0,0.85,0.05)    # volatility of stock

PV_K = K * np.exp(-r*T);

C = []

for value in sigma:
    d1 = (np.log(S/K) + (r + (value**2/2))*T)/(value*np.sqrt(T))
    d2 = d1 - value * np.sqrt(T)
    C.append(norm.cdf(d1)*S - norm.cdf(d2)*PV_K)


plt.plot(sigma,C,'o-')
plt.xlabel('Volatility')
plt.ylabel('Call Option Value')
plt.title('Option Value vs. Volatility')
plt.show()

S = 100                             # current stock price (spot price)
K = 100                             # exercise price (strike price)
sigma = 0.2                         # volatility of stock
r = 0.05                            # annualized risk-free interest rate 

T = np.arange(1,13,1)/12               # expiry time in years

C = []


for value in T:
    PV_K = K * np.exp(-r*value);
    d1 = (np.log(S/K) + (r + (sigma**2/2))*value)/(sigma*np.sqrt(value))
    d2 = d1 - sigma * np.sqrt(value)
    C.append(norm.cdf(d1)*S - norm.cdf(d2)*PV_K)


plt.plot(T,C,'o-')
plt.xlabel('Time to Expiry')
plt.ylabel('Call Option Value')
plt.title('Option Value vs. Time to Expiry')
plt.show()