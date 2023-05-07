import numpy as np


def counting(n, x):
    a = np.zeros((n, x))
    for i in range(0, min(n, x)):
        a[i][i] = 1
    for i in range(x):
        a[0][i] = 1
    for j in range(1, x):
        for i in range(1, min(j, n)):
            a[i][j] = a[i-1][j-1] + a[i][j-1]

    print(f'PSI_N(X) = {a[n-1][x-1]}')


counting(12, 800)
