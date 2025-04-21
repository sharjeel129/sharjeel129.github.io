import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.utils.data
from torchvision import datasets, transforms

def Energy(v, h, W, a, b):
    E = -sum(v[i] * W[i][j] * h[j] for i in range(len(v)) for j in range(len(h)))
    E -= sum(a[i] * v[i] for i in range(len(v)))
    E -= sum(b[j] * h[j] for j in range(len(h)))
    return E

# def objective(p, q):
#     O = 0
#     for i in range(len(p)):
#         O += q[i]*np.log(q[i]/p[i])
#     return O

def eff_mag_field(v, W, b):
    Nv = np.shape(W)[0]
    Nh = np.shape(W)[1]

    if len(v) == Nv:
        m = np.zeros(Nh)
        for i in range(Nv):
            m += v[i]*W[i, :]
    else:
        m = np.zeros(Nv)
        for i in range(Nh):
            m += W[:, i]*v[i]

    return m + b


def Z_Prob(v, h, W, a, b):
    Nv, Nh = len(v), len(h)
    
    prob = sum(
        np.exp(-Energy(
            np.array([1 if format(i, f'0{Nv}b')[k] == '1' else -1 for k in range(Nv)]),
            np.array([1 if format(j, f'0{Nh}b')[k] == '1' else -1 for k in range(Nh)]),
            W, a, b
        ))
        for i in range(2**Nv) for j in range(2**Nh)
    )
    
    return prob

def cond_Prob(v, h, W, a, b):
    prob = 1
    m = eff_mag_field(h, W, a)
    N = len(v)

    for i in range(N):
        prob *= 1 / (1 + np.exp(-2 * v[i] * m[i]))

    return prob

def full_Prob(v, h, W, a, b):
    Z = Z_Prob(v, h, W, a, b)
    Nv, Nh = len(v), len(h)
    probs_theory = np.zeros((2**Nv, 2**Nh))
    
    for i in range(2**Nv):
        v = [1 if format(i, f'0{Nv}b')[k] == '1' else -1 for k in range(Nv)]
        for j in range(2**Nh):
            h = [1 if format(j, f'0{Nh}b')[k] == '1' else -1 for k in range(Nh)]
            probs_theory[i][j] = np.exp(-Energy(v, h, W, a, b)) / Z
    
    return probs_theory, np.sum(probs_theory, axis=1), np.sum(probs_theory, axis=0)


def sampler(v, h, W, a, b):
    m = eff_mag_field(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = 1 if np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i])) > np.random.rand() else -1

    return newV

def sampler_MNIST(v, h, W, a, b):
    m = eff_mag_field(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = 1 if np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i])) > np.random.rand() else 0

    return newV

def prob_sampler_MNIST(v, h, W, a, b):
    m = eff_mag_field(h, W, a)

    N = len(v)

    newV = np.zeros(N)

    for i in range(N):
        newV[i] = np.exp(m[i])/(np.exp(m[i]) + np.exp(-m[i]))

    return newV

def get_Rand_RBM(Nv, Nh):
    v = np.random.choice([1,-1], Nv)
    h = np.random.choice([1,-1], Nh)

    a = (np.random.rand(Nv)*2)-1
    b = (np.random.rand(Nh)*2)-1

    W = (np.random.rand(Nv,Nh)*2)-1

    return v,h,W,a,b


def get_Rand_RBM_MNIST(Nv, Nh):
    v = np.random.choice([0,1], Nv)
    h = np.random.choice([0,1], Nh)

    a = (np.random.rand(Nv)*2)-1
    b = (np.zeros(Nh))

    W = (np.random.rand(Nv,Nh)*2*0.01)-0.01

    return v,h,W,a,b


def evolver(v, h, W, a, b, k):
    for i in range(k):
        v = sampler(v, h, W, a, b)
        h = sampler(h, v, W, b, a)
    return v, h

def evolver_MNIST(v, h, W, a, b, k):
    for i in range(k):
        v = sampler_MNIST(v, h, W, a, b)
        h = sampler_MNIST(h, v, W, b, a)
    return v, h

def free_energy(v, h, W, a, b):
    num_hidden = len(h)
    total = 0

    for i in range(2**num_hidden):
        binary_state = format(i, f"0{num_hidden}b")
        h_state = np.array([1 if bit == "1" else -1 for bit in binary_state])
        total += np.exp(-Energy(v, h_state, W, a, b))

    return -np.log(total)

def free_energy_alt(v, h, W, a, b):
    v_dot_a = np.dot(v, a)
    W_transpose_v = np.dot(W.T, v)
    log_term = np.log(np.exp(-b - W_transpose_v) + np.exp(b + W_transpose_v))
    return -v_dot_a - np.sum(log_term)

def free_energy_MNIST(v, h, W, a, b):
    v_dot_a = np.dot(v, a)
    W_transpose_v = np.dot(W.T, v)
    exp_term = np.exp(b + W_transpose_v)
    log_term = np.log(1 + exp_term)
    return -v_dot_a - np.sum(log_term)

def objective(p, q):
    result = 0
    for i in range(len(p)):
        ratio = q[i] / p[i]
        result += q[i] * np.log(ratio)
    return result

def PlotMe(data):
    n = 8 
    
    save_plot = np.zeros((1, 1))
    to_plot = np.zeros((1, 1))

    for i in range(n):
        for j in range(n):
            idx = i * n + j 

            image = data[idx, :].reshape((28, 28))

            if j == 0:
                to_plot = image
            else:
                to_plot = np.hstack((to_plot, image))

        if i == 0:
            save_plot = to_plot.copy()
        else:
            save_plot = np.vstack((save_plot, to_plot))

    return save_plot
  

# Nv = 5
# Nh = 2

# k = 20

# v, h, W, a, b = get_Rand_RBM(Nv, Nh)

# count = 100000

# P(h|v)

# probs_sampling = np.zeros(2**Nh)

# for i in range(count):
#     h = sampler(h, v, W, b, a)
#     probs_sampling[int("".join(["1" if h[i] == 1 else "0" for i in range(len(h))]), 2)] += 1
    
# probs_sampling /= count

# probs_theory = np.zeros(2**Nh)

# for i in range(2**Nh):
#     h = [1 if format(i, f"0{Nh}b")[j] == "1" else -1 for j in range(Nh)]
#     probs_theory[i] = cond_Prob(h, v, W, b, a)

# bins = np.arange(2 ** Nh)
# plt.figure(figsize=(8, 6))
# plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampled')
# plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theoretical')
# plt.xticks(bins, [i for i in range(2 ** Nh)])
# plt.xlabel('Hidden States')
# plt.ylabel('Probability')
# plt.title('P(h|v) Sampled vs. Theoretical Probability Distribution')
# plt.legend()
# plt.show()

# print("Sampling Probabilities:", probs_sampling)
# print("Theoretical Probabilities:", probs_theory)
# print("Sum of Theoretical Probabilities:", probs_theory.sum())

# P(v|h)

# probs_sampling = np.zeros(2**Nv)

# for i in range(count):
#     v = sampler(v,h,W,a,b)
#     probs_sampling[int("".join(["1" if v[i] == 1 else "0" for i in range(len(v))]), 2)] += 1

# probs_sampling /= count

# probs_theory = np.zeros(2**Nv)

# for i in range(2**Nv):
#     v = [1 if format(i, "0" + str(Nv) + 'b')[j] == "1" else -1 for j in range(Nv)]
#     probs_theory[i] = cond_Prob(v, h, W, a, b)


# bins = np.arange(2 ** Nv)
# plt.figure(figsize=(8, 6))
# plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampled')
# plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theoretical')
# plt.xticks(bins, [i for i in range(2 ** Nv)])
# plt.xlabel('Visible States')
# plt.ylabel('Probability')
# plt.title('P(v|h) Sampled vs. Theoretical Probability Distribution')
# plt.legend()
# plt.show()

# print("Sampling Probabilities:", probs_sampling)
# print("Theoretical Probabilities:", probs_theory)
# print("Sum of Theoretical Probabilities:", probs_theory.sum())

# P(v,h)

# probs_sampling = np.zeros((2**Nv, 2**Nh))

# for _ in range(count):
#     vf, hf = evolver(v, h, W, a, b, k)
#     v_idx = int("".join("1" if bit == 1 else "0" for bit in vf), 2)
#     h_idx = int("".join("1" if bit == 1 else "0" for bit in hf), 2)
#     probs_sampling[v_idx, h_idx] += 1

# probs_sampling /= count

# probs_theory, _, _ = full_Prob(v, h, W, a, b)

# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# x, y = np.meshgrid(np.arange(2**Nv), np.arange(2**Nh))
# x, y = x.flatten(), y.flatten()
# z = np.zeros_like(x)

# bar_width, bar_depth = 0.4, 0.8

# ax.bar3d(x - bar_width/2, y, z, bar_width, bar_depth, probs_sampling.flatten(), 
#           color='b', alpha=0.8, label='Sampled')

# ax.bar3d(x + bar_width/2, y, z, bar_width, bar_depth, probs_theory.flatten(), 
#           color='r', alpha=0.8, label='Theoretical')

# ax.set_xlabel('Visible States')
# ax.set_ylabel('Hidden States')
# ax.set_zlabel('Probability')
# ax.set_title('P(v,h) Sampled vs. Theoretical Probabilities')
# ax.set_xticks(np.arange(2**Nv))
# ax.set_yticks(np.arange(2**Nh))
# ax.legend()
# plt.tight_layout()
# plt.show()

# print("Sampling Probabilities:\n", probs_sampling)
# print("\nTheoretical Probabilities:\n", probs_theory)
# print("\nSum of Theoretical Probabilities:", probs_theory.sum())

# P(v)

# probs_sampling = np.zeros((2**Nv, 2**Nh))

# for _ in range(count):
#     vf, hf = evolver(v, h, W, a, b, k)
#     v_idx = int("".join("1" if bit == 1 else "0" for bit in vf), 2)
#     h_idx = int("".join("1" if bit == 1 else "0" for bit in hf), 2)
#     probs_sampling[v_idx, h_idx] += 1

# probs_sampling /= count

# mat_sampling = probs_sampling.copy()
# mat_theory = probs_sampling.copy()

# probs_sampling = np.sum(mat_sampling, axis=1)

# probs_theory = np.sum(mat_theory, axis=1)

# bins = np.arange(2 ** Nv)
# plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampled')
# plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theoretical')
# plt.xticks(bins, [i for i in range(2 ** Nv)])
# plt.xlabel('Visible States')
# plt.ylabel('Probability')
# plt.title('P(v) Sampled vs. Theoretical Probability Distribution')
# plt.legend()
# plt.show()

# P(h)

# probs_sampling = np.zeros((2**Nv, 2**Nh))

# for _ in range(count):
#     vf, hf = evolver(v, h, W, a, b, k)
#     v_idx = int("".join("1" if bit == 1 else "0" for bit in vf), 2)
#     h_idx = int("".join("1" if bit == 1 else "0" for bit in hf), 2)
#     probs_sampling[v_idx, h_idx] += 1

# probs_sampling /= count

# mat_sampling = probs_sampling.copy()
# mat_theory = probs_sampling.copy()

# probs_sampling = np.sum(mat_sampling, axis=0)

# probs_theory = np.sum(mat_theory, axis=0)

# bins = np.arange(2 ** Nh)
# plt.bar(bins - 0.2, probs_sampling, width=0.4, label='Sampled')
# plt.bar(bins + 0.2, probs_theory, width=0.4, label='Theoretical')
# plt.xticks(bins, [i for i in range(2 ** Nh)])
# plt.xlabel('Hidden States')
# plt.ylabel('Probability')
# plt.title('P(h) Sampled vs. Theoretical Probability Distribution')
# plt.legend()
# plt.show()

# print("Sampling Probabilities:", probs_sampling)
# print("Theoretical Probabilities:", probs_theory)
# print("Sum of Theoretical Probabilities:", probs_theory.sum())

# Training

# Nv = 5
# Nh = 3

# M = 64
# k = 1
# eta = 0.1

# prob_dist = np.random.ranf(2**Nv)
# prob_dist = prob_dist/np.sum(prob_dist)
# data = np.random.choice(range(0,2**Nv),p=prob_dist,size=100000)

# v,h,W,a,b = get_Rand_RBM(Nv, Nh)
# count = 50

# counter = 0

# o_list = []
# f_list = []

# for i in range(count):
#     f_list.append(free_energy(v, h, W, a, b))

#     # Initialize gradients
#     dW = np.zeros((Nv, Nh))
#     da = np.zeros(Nv)
#     db = np.zeros(Nh)

#     if (counter + 1) * M > len(data):
#         np.random.shuffle(data)
#         counter = 0

#     batch_data = data[counter * M : (counter + 1) * M]

#     for j in range(M):
#         bits = format(batch_data[j], "0" + str(Nv) + 'b')
#         v = np.array([1 if bits[m] == "1" else -1 for m in range(Nv)])

#         h = sampler(h, v, W, b, a)

#         dW -= np.outer(v, h)
#         da -= v
#         db -= h

#         v, h = evolver(v, h, W, a, b, k)

#         dW += np.outer(v, h)
#         da += v
#         db += h

#     W -= eta * dW / M
#     a -= eta * da / M
#     b -= eta * db / M

#     counter += 1

# mat, v_dist, h_dist = full_Prob(v, h, W, a, b)

# plt.figure(0)
# plt.plot(prob_dist)
# plt.plot(v_dist)
# plt.xlabel("Configuration")
# plt.ylabel("Probability")
# plt.title("RBM Model Learning Probability Distribution")
# plt.legend(("Data", "RBM"))

# F_derv = np.diff(f_list)

# plt.figure(1)
# plt.plot(F_derv)
# plt.xlabel("Epoch")
# plt.ylabel("Delta Free Energy")
# plt.title("Free Energy")
# plt.show()

# MNIST

batch_size = 1
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)
myData=[]
for idx, (data,target) in enumerate(train_loader):
  myData.append(np.array(data.view(-1,784)).flatten())
myData=np.matrix(myData)

# pic=np.copy(myData[0,:])
# pic=pic.reshape((28,28))
# plt.matshow(pic)
# plt.show()

# plt.matshow(PlotMe(myData[0:64,:]))

Nv = 784
Nh = 400

M = 64
k = 1
eta = 0.1

v,h,W,a,b = get_Rand_RBM_MNIST(Nv, Nh)
count = 200

counter = 0

for i in range(count):

    # Initialize gradients
    dW = np.zeros((Nv, Nh))
    da = np.zeros(Nv)
    db = np.zeros(Nh)

    # Shuffle data if needed
    if (counter + 1) * M > len(myData):
        np.random.shuffle(myData)
        counter = 0

    # Get mini-batch
    batch_data = myData[counter * M : (counter + 1) * M]

    for j in range(M):

        # Sample visible layer from batch with noise
        v = np.array([
            1 if (batch_data[j, :])[0, m] > np.random.rand() else 0 
            for m in range(Nv)
        ])

        # Sample hidden units
        h = sampler_MNIST(h, v, W, b, a)

        # Negative phase
        dW -= np.outer(v, h)
        da -= v
        db -= h

        # Contrastive Divergence step
        v, h = evolver_MNIST(v, h, W, a, b, k)

        # Positive phase
        dW += np.outer(v, h)
        da += v
        db += h

    # Update weights and biases
    W -= eta * dW / M
    a -= eta * da / M
    b -= eta * db / M

    counter += 1


visible = np.copy(myData[0:M, :])
plt.matshow(PlotMe(visible))

# Sample reconstructions using sampler_MNIST twice
vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()
    h = sampler_MNIST(h, v2, W, b, a)
    v2 = sampler_MNIST(v2, h, W, a, b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

plt.matshow(PlotMe(vst))

# Sample reconstructions using prob_sampler_MNIST
vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()
    h = sampler_MNIST(h, v2, W, b, a)
    v2 = prob_sampler_MNIST(v2, h, W, a, b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

plt.matshow(PlotMe(vst))

# Long chain sampling with k steps
vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()

    k = 100
    for j in range(k - 1):
        h = sampler_MNIST(h, v2, W, b, a)
        v2 = sampler_MNIST(v2, h, W, a, b)

    h = sampler_MNIST(h, v2, W, b, a)
    v2 = prob_sampler_MNIST(v2, h, W, a, b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

plt.matshow(PlotMe(vst))

# Add dropout-style noise to visible input
mask = np.random.rand(*visible.shape) > 0.5
visible[mask] = 0
plt.matshow(PlotMe(visible))

# Try to reconstruct the noisy input
vst = None
for i in range(M):
    v2 = np.array(visible[i, :]).flatten()
    h = sampler_MNIST(h, v2, W, b, a)
    v2 = prob_sampler_MNIST(v2, h, W, a, b)

    if i == 0:
        vst = v2.copy()
    else:
        vst = np.vstack((vst, v2))

plt.matshow(PlotMe(vst))
plt.show()

# Weights

W = np.load("Weights.npy")
print(W.transpose().shape)
plt.matshow(PlotMe(W.transpose()))

# Discrimination

batch_size = 1
train_loader = torch.utils.data.DataLoader(
datasets.MNIST('./data',
    train=True,
    download = True,
    transform = transforms.Compose(
        [transforms.ToTensor()])
     ),
     batch_size=batch_size
)
myData=[]
for idx, (data,target) in enumerate(train_loader):
  myData.append(np.array(data.view(-1,784)).flatten())

myData=np.matrix(myData)

h = np.load("h.npy")
W = np.load("Weights.npy")
a = np.load("a.npy")
b = np.load("b.npy")

v = np.array(myData[0, :]).flatten()

print(free_energy_MNIST(v,h,W,a,b))

v = np.random.choice([1, 0], len(v))

print(free_energy_MNIST(v,h,W,a,b))