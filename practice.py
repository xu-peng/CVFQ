import numpy as np
myEsti = np.genfromtxt('./data/estimated.csv', delimiter=',')

# Generate Data
np.random.seed(5)
P_matrix = np.random.choice(a=[False, True], size=[400, 3], p=[0.5, 0.5])
Q_matrix = np.random.choice(a=[False, True], size=[3, 9], p=[0.5, 0.5])
R_matrix = np.matmul(P_matrix, Q_matrix).astype(float)

R_matrix2 = np.genfromtxt('./data/R_matrix.csv', delimiter=',')


