import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import multivariate_normal, wishart
from scipy.special import digamma
from mpl_toolkits.mplot3d import Axes3D
import sklearn.decomposition as deco
from database_actions import get_known_encodings

## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
    l, V = np.linalg.eigh(Sigma)
    l[l<0] = 0
    t = np.linspace(0.0, 2.0*np.pi, 100)
    xy = np.stack((np.cos(t), np.sin(t)))
    Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
    plt.plot(Txy[:, 0], Txy[:, 1])

def logdet(Sigma):
    (s, ulogdet) = np.linalg.slogdet(Sigma)
    return s*ulogdet

## Load data
data = get_known_encodings()[0].transpose()
x = (data - np.mean(data, 0)) / np.std(data, 0)
pca = deco.PCA(2)
data = pca.fit(x).transform(x)
N, D = data.shape

## Number of components/clusters
K = 5

## Priors
alpha0 = 1e-3 # Mixing prior (small number: let the data speak)
m0 = np.zeros(D); beta0 = 1e-3 # Gaussian mean prior
v0 = 3e1; W0 = np.eye(D)/v0 # Wishart covariance prior

## Initialize parameters
m_k = []
W_k = []
beta_k = np.repeat(beta0 + N / K, K)
alpha_k = np.repeat(alpha0 + N / K, K)
v_k = np.repeat(v0 + N / K, K)
for _ in range(K):
    # Let m_k be a random data point:
    m_k.append(data[np.random.choice(N)])
    # Let W_k be the mean of the Wishart prior:
    W_k.append(v0 * W0)

## Loop until you're happy
max_iter = 1000
ln_rho = np.zeros((N, K))
for iteration in range(max_iter):

    ## Variational E-step
    Elnpi = digamma(alpha_k) - digamma(np.sum(alpha_k))
    for k in range(K):
        delta_k = data - m_k[k]  # NxD
        EmuL = D / beta_k[k] + v_k[k] * np.sum(delta_k.dot(W_k[k]) * delta_k, axis=1)  # Nx1
        ElnL = np.sum(digamma(0.5 * (v_k[k] - np.arange(D)))) + D * np.log(2) + logdet(W_k[k])  # 1x1
        ln_rho[:, k] = Elnpi[k] + 0.5 * ElnL - 0.5 * D * np.log(2 * np.pi) - 0.5 * EmuL  # Nx1

    rho = np.exp(ln_rho - np.max(ln_rho, axis=1).reshape((N, 1)))  # NxK
    r_nk = rho / np.sum(rho, axis=1).reshape((N, 1))  # NxK

    ## Variational M-step
    Nk = np.sum(r_nk, axis=0)  # 1xK
    alpha_k = alpha0 + Nk  # 1xK
    beta_k = beta0 + Nk  # 1xK
    v_k = v0 + Nk  # 1xK
    for k in range(K):
        rk = r_nk[:, k] / Nk[k]  # Nx1
        rk[Nk[k] < 1e-6] = 0
        xbar = rk.T.dot(data)
        delta_k = data - xbar  # NxD
        Sk = delta_k.T.dot(np.diag(rk)).dot(delta_k)  # DxD

        m_k[k] = (beta0 * m0 + Nk[k] * xbar) / beta_k[k]
        Winv = np.linalg.inv(W0) + Nk[k] * Sk + (beta0 * Nk[k] / (beta0 + Nk[k])) * (xbar - m0).T.dot(xbar - m0)
        W_k[k] = np.linalg.pinv(Winv)

## Plot data with distribution (we show expected distribution)
plt.figure(1)
plt.plot(data[:, 0], data[:, 1], '.')
for k in range(K):
    if Nk[k] > 0:
        plot_normal(m_k[k], np.linalg.pinv(v_k[k] * (W_k[k])))
plt.show()

## Animate the entire mixture distribution
fig = plt.figure(2)
num_samples = 100
ax = fig.add_subplot(111, projection='3d')
lim = np.linspace(-10, 10, 20)
X, Y = np.meshgrid(lim, lim)
XY = np.vstack((X.flatten(), Y.flatten())).T
for s in range(num_samples):
    pi_k = np.random.dirichlet(alpha_k)
    Z = np.zeros(X.shape).flatten()
    for k in range(K):
        L = wishart.rvs(scale=W_k[k], df=v_k[k])
        Sigma = np.linalg.pinv(L)
        mu = multivariate_normal.rvs(mean=m_k[k], cov=Sigma / beta_k[k])
        Z += pi_k[k] * multivariate_normal.pdf(XY, mean=mu, cov=Sigma)
    ax.cla()
    ax.plot_surface(X, Y, Z.reshape(X.shape))
    ax.set_zlim(top=0.05)
    plt.pause(0.01)