from math import sqrt
import numpy as np
from lab1 import find_solution, draw_table


def octahedral_norm(matrix):
    return np.max([np.sum(np.abs(row)) for row in np.array(matrix)])

def cubic_norm(matrix):
    return octahedral_norm(np.array(matrix).T)

def find_inverse_matrix(A):
    A = np.copy(A)
    inverseA = np.eye(len(A))
    for j in range(len(A) - 1):
        for i in range(j + 1, len(A)):
            koef = -A[i][j] / A[j][j]
            A[i] = A[j] * koef + A[i]
            inverseA[i] = inverseA[j] * koef + inverseA[i]
    for i in range(len(A) - 1, -1, - 1):
        inverseA[i] = np.array(inverseA[i]) / A[i][i]
        A[i] = np.array(A[i]) / A[i][i]
        for j in range(i - 1, -1, -1):
            inverseA[j] = list(np.array(inverseA[j]) - np.array(inverseA[i]) * A[j][i])
            A[j] = list(np.array(A[j]) - np.array(A[i]) * A[j][i])
    return inverseA

if __name__ == "__main__":
    D = np.array([[6.22, 1.42, -1.72, 1.91],
                  [1.44, 5.33, 1.11, -1.82],
                  [-1.72, 1.11, 5.24, 1.42],
                  [1.91, -1.82, 1.42, 6.55]])
    k = 2
    C = np.eye(4)
    A = D + k*C
    # inv_A = find_inverse_matrix(A)
    inv_A = [[],[],[],[]]
    print(f"matrix A:\n{A}")
    inv_A[0], _ = find_solution(A, np.array([1,0,0,0]))
    draw_table()
    inv_A[1], _ = find_solution(A, np.array([0,1,0,0]))
    draw_table()
    inv_A[2], _ = find_solution(A, np.array([0,0,1,0]))
    draw_table()
    inv_A[3], _ = find_solution(A, np.array([0,0,0,1]))
    draw_table()
    inv_A = np.array(inv_A)
    print(f"inverse matrix for find A:\n{inv_A}")
    print(f"A * A^-1 == E:\n{np.allclose(np.eye(4), inv_A.dot(A), atol=0.0001)}")
    print(f"first condition {octahedral_norm(A) * octahedral_norm(inv_A)}")
    print(f"first condition {cubic_norm(A) * cubic_norm(inv_A)}")