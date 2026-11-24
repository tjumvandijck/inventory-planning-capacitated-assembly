import numpy as np
from scipy.optimize import linprog


def find_probabilities(x, mu1, CV0):
    x = np.asarray(x, dtype=float)
    n = len(x)
    M2 = (1.0 + CV0**2) * (mu1**2)
    print(M2)
    # quick mean feasibility check
    if not (x.min() - 1e-12 <= mu1 <= x.max() + 1e-12):
        return None, "Target mean outside support range"

    # objective zero (we only want a feasible point)
    c = np.zeros(n)

    # Equality constraints A_eq @ p = b_eq
    A_eq = np.vstack([np.ones(n), x**2])
    b_eq = np.array([1.0, M2])

    # bounds p_i >= 0
    bounds = [(0.01, None)] * n
    bounds[0] = (0.05, 0.05)
    bounds[2] = (None, 0.6)
    bounds[-1] = (0.005, 0.005)
    bounds[-2] = (0.01, None)

    # Inequality constraint: p0 + p1 >= p3 + p4 + p5
    A_ub = np.ones((3, n))
    A_ub[0, 0] = -1   # -p0
    A_ub[0, 1] = -1   # -p1
    A_ub[0, 2] =  0   # +p2
    A_ub[0, 3] =  1   # +p3
    A_ub[0, 4] =  1   # +p4
    A_ub[0, 5] =  1   # +p5

    b_ub = np.array([-0.01, 0.9*mu1, 1.1*mu1])  # Right-hand side

    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  A_ub=A_ub, b_ub=b_ub,
                  bounds=bounds, method='highs')

    if res.success:
        p = res.x
        # numerical cleanup
        p[p < 1e-12] = 0.0
        p = p / p.sum()
        return p, None
    else:
        return None, "Infeasible: cannot meet mean & CV with given support"



# Example usage
x = np.array([2, 3, 4, 5, 6, 7])  # support (6 elements)
prob_other_vec = np.array([0.04, 0.18, 0.21, 0.23, 0.14, 0.09, 0.07, 0.04])
elements_other = np.array([7,8,9,10,11,12,13,14])
mu0 = 10

# normalize in case p doesn't sum exactly to 1
p = prob_other_vec / prob_other_vec.sum()

# weighted mean
mu = np.sum(p * elements_other)

# weighted variance
var = np.sum(p * (elements_other - mu)**2)

# standard deviation
std0 = np.sqrt(var)
print("Mean, var, CV:", mu, var, std0/mu)

CV0 = std0 / mu0
mu1 = 4

p, err = find_probabilities(x, mu1, CV0)
if p is None:
    print("No solution:", err)
else:
    print("Found probabilities:", p)
    print("Check mean, var, cv:", np.dot(p, x), np.dot(p, x**2) - np.dot(p, x)**2, np.sqrt(np.dot(p, x**2) - np.dot(p, x)**2) / np.dot(p, x))
a = 1