import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from database_actions import get_known_encodings
import sklearn.decomposition as deco

## Helper function for plotting a 2D Gaussian
def plot_normal(mu, Sigma):
    l, V = np.linalg.eigh(Sigma)
    l[l < 0] = 0
    t = np.linspace(0.0, 2.0 * np.pi, 100)
    xy = np.stack((np.cos(t), np.sin(t)))
    Txy = mu + ((V * np.sqrt(l)).dot(xy)).T
    plt.plot(Txy[:, 0], Txy[:, 1])


## Load data
data = get_known_encodings()[0].transpose()
x = (data - np.mean(data, 0)) / np.std(data, 0)
pca = deco.PCA(2)
data = pca.fit(x).transform(x)
N, D = data.shape

## Initialize parameters
K = 5; # clusters
mu = []
Sigma = []
pi_k = np.ones(K)/K
for _ in range(K):
  # Let mu_k be a random data point:
  mu.append(data[np.random.choice(N)])
  # Let Sigma_k be the identity matrix:
  Sigma.append(np.eye(D))

## Loop until you're happy
max_iter = 1000;
log_likelihood = np.zeros(max_iter)
respons = np.zeros((K, N)) # KxN
for iteration in range(max_iter):
  ## Compute responsibilities
  for k in range(K):
    respons[k] = pi_k[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
  respons /= np.sum(respons, axis=0)

  ## Update parameters
  for k in range(K):
      respons_k = respons[k]  # N
      Nk = np.sum(respons_k)  # scalar
      mu[k] = respons_k.dot(data) / Nk  # D
      delta = data - mu[k]  # NxD
      Sigma[k] = (respons_k * delta.T).dot(delta) / Nk  # DxD
      pi_k[k] = Nk / N

  ## Compute log-likelihood of data
  L = 0
  for k in range(K):
      L += pi_k[k] * multivariate_normal.pdf(data, mean=mu[k], cov=Sigma[k])
  log_likelihood[iteration] = np.sum(np.log(L))

print log_likelihood

## Plot log-likelihood -- did we converge?
plt.figure(1)
plt.plot(log_likelihood)
plt.xlabel('Iterations')
plt.ylabel('Log-likelihood')
plt.savefig("figures/em_loglikelihood")

## Plot data
plt.figure(2)
plt.plot(data[:, 0], data[:, 1], '.')

for k in range(K):
    plot_normal(mu[k], Sigma[k])

plt.savefig("figures/em_algorithm_clusters")
plt.show()