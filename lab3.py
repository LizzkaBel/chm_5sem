from math import sqrt
import numpy as np

def cubic_r(vector):
    return max([abs(x) for x in vector])

def octahedral_r(vector):
    return sum([abs(x) for x in vector])

def spherical_r(vector):
    return sqrt(sum([x * x for x in vector]))


def find_solution(A, f):
    A = list(A)
    f = list(f)
    for j in range(len(A) - 1):
        for i in range(j + 1, len(A)):
            koef = -A[i][j] / A[j][j]
            A[i] = list(np.array(A[j]) * koef + A[i])
            f[i] = f[j] * koef + f[i]
    solutions = []
    for i in range(len(A) - 1, -1, -1):
        temp = f[i]
        for j in range(1, len(A) - i):
            temp -= A[i][len(A) - j] * solutions[j - 1]
        temp /= A[i][i]
        solutions.append(temp)
    solutions.reverse()
    return solutions


def find_L(A):
    A = np.copy(A)
    L = np.zeros((len(A), len(A)))   
    for k in range(len(A)):
        L[k][k] = sqrt(A[k][k] - sum([L[k][x]**2 for x in range(k)]))
        for i in range(k + 1, len(A)):
            L[i][k] = (A[i][k] - sum([L[i][x]*L[k][x] for x in range(k)]))/L[k][k] 
    return L
    

if __name__ == "__main__":
    A = np.array([
        [1.65, -1.76, 0.77],
        [-1.76, 1.04, -2.61],
        [0.77, -2.61, -3.18]
        ])
    f = np.array([2.15, 0.82, -0.73])
    print(f"A == A^T {np.array_equal(A, np.transpose(A))}")
    print(f"A is positive defined: {np.all(np.linalg.eigvals(A) > 0)}")
    B = A.dot(A)
    L = find_L(B)
    print(f"L is:\n {L}")
    print(f"B = L*L^T {np.allclose(B, L.dot(L.transpose()), atol=0.0001)}") 
    y = find_solution(L, A.dot(f))
    x = find_solution(L.transpose(), y)
    print(f"my x* is {x}")
    r = A.dot(x) - f
    print(f"cubic norm of vector {cubic_r(r)}")
    print(f"octahedral norm of vector {octahedral_r(r)}")
    print(f"spherical norm of vector {spherical_r(r)}")