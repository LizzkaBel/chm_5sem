from math import sqrt
import numpy as np

def octahedral_norm(matrix):
    return np.max([np.sum(np.abs(row)) for row in np.array(matrix)])

def check_condition(A):
    for i in range(len(A)):
        print(A[i][i] > sum([abs(A[i][j]) for j in range(len(A)) if i != j]))



if __name__ == "__main__":
    k = 8
    A = np.array([
        [2, 1, 0.1 * k],
        [0.1 * k, 5, 0.72],
        [-1.2, 3, 1.7]      # 0.9 > -3.2 + 2 
    ])
    f = np.array([-2.9, -0.7, -9.86])
    print(np.linalg.solve(A,f))
    A[2] = A[2] - A[1] + A[0] 
    f[2] = f[2] - f[1] + f[0]
    check_condition(A)
    print(A)
    E = 0.5/(10**4)
    D = np.zeros((3,3))
    for i in range(3):
        D[i][i] = A[i][i]
    C = np.linalg.inv(D)
    B = np.eye(3) - C.dot(A)
    print(f"cubic norm b: {octahedral_norm(B)}")
    g = C.dot(f)
    x = [g]
    E1 = (1 - octahedral_norm(B))/octahedral_norm(B) * E
    while True:
        x.append(B.dot(x[-1]) + g)
        if octahedral_norm(x[-1] - x[-2]) <= E1:
            break
    print(len(x), x[-1])
    # print(f"A == A^T {np.array_equal(A, np.transpose(A))}")
    # B = A.dot(A)
    # print(f"B = L*L^T {np.allclose(B, L.dot(L.transpose()), atol=0.0001)}") 