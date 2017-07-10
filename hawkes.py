import numpy as np

# Hawkes EM Algorithm initial implementation

## -----------------------------
## Data provided

K = 4
N = 1000

a = np.random.rand(N)
sn = np.cumsum(a)
cn = np.random.choice(range(K), size=N)

tmax = np.zeros(K)
for k in range(K):
    tmax[k] = np.max(sn[cn == k])

# initialize the "parameters"

mu  = np.random.rand(K)
lda = np.random.rand(K)
Phi = np.random.rand(K, K)

# freeze some of the values for repeated use
cd = cdist(sn[:, np.newaxis], sn[:, np.newaxis], metric="cityblock")


for epoch in range(100):
    # E-step
    zeta  = np.log(lda[cn])
    Gamma = np.zeros((N, N))

    Gamma = (Gamma.T + np.log(mu[cn])).T
    Gamma -= (cd.T * mu[cn]).T
    
    for i in range(1,N):
        for j in range(i):
#             Gamma[i, j] += np.log(mu[cn[i]])
#             Gamma[i, j] -= cd[i, j] * mu[cn[i]]
            Gamma[i, j] += np.log(Phi[cn[j], cn[i]])

    zeta[0] = 0
    for i in range(1,N):
        g = Gamma[i,:i]
        lsexp = np.logaddexp(zeta[i], logsumexp(g))
        Gamma[i,:i] -= lsexp
        zeta[i] -= lsexp
        
#     print(np.exp(zeta[5]) + np.exp(Gamma[5, :5]).sum())

    # M-step
    G = np.exp(Gamma)
    G *= np.tri(*G.shape)
    np.fill_diagonal(G, 0)
    
#     print np.exp(zeta) + np.sum(G, 1)

    for k in range(K):
        lda[k] = np.sum(np.exp(zeta)[cn==k]) / tmax[k]

        mu[k]  = G[cn == k, :].sum() / (G * cd)[cn == k, :].sum() 

    for a in range(K):
        for b in range(K):        
            Phi[a, b] = G[cn==b, :][:, cn==a].sum() / np.sum((cn == a)[:-1])

#     print lda, mu, Phi

    # calculate the ECDLL
    
    res = np.sum(-lda * tmax) + np.sum(np.exp(zeta) * np.log(lda)[cn]) - np.sum(Phi.T * np.bincount(cn[:-1]))
    for a in range(K):
        for b in range(K):
            res += np.log(Phi)[a, b] * G[cn==b, :][:, cn==a].sum()

    for a in range(K):
        res += np.log(mu[a]) * G[cn == a, :].sum()

    for a in range(K):
        res -= mu[a] * (G * cd)[cn == a, :].sum()
        
    print res, np.sum(-lda * tmax) + np.sum(np.exp(zeta) * np.log(lda)[cn]), np.linalg.norm(Phi, "fro")
