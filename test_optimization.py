import numpy as np
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas

# Generate random vector r and symmetric definite positive matrix Q
n = 50
r = matrix(np.random.sample(n))
Q = np.random.randn(n,n)
Q = 0.5 * (Q + Q.T)
Q = Q + n * np.eye(n)
Q = matrix(Q)
# Solve
sol = qp(Q, -r)
print(sol)