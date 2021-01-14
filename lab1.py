from math import sqrt
import numpy as np
import matplotlib.pyplot as plt



data = []

def draw_table():
	global data
	column_headers = ["x" + str(x) for x in range(len(data[0]) - 3)] + ["f", "Sum1", "Sum2"]
	cell_text = []
	for rows in data:
		cell_text.append([])
		for el in rows:
			if type(el) == list:
				out_string = ",".join([f"{x:.4f}" if type(x) != int else str(x) for x in el])
			elif type(el) == str:
				out_string = el
			else:
				out_string = f"{el:.4f}"
			cell_text[-1].append(out_string)
	ax = plt.subplot2grid((1,1), (0,0))
	ax.table(cellText=cell_text, colLabels=column_headers, loc='center', fontsize=100)
	ax.axis("off")
	plt.show()

def append_data_to_table(*args):
	data.append([])
	for i in args:
		# print(i, end= "\n" if i == args[-1] else ", ")
		data[-1].append(i)

def find_solution(A, f):
	A = list(A)
	f = list(f)
	s = []
	for j in range(0, len(A) - 1):
		append_data_to_table(*A[j], f[j], sum(A[j]) + f[j], s.pop(0) if j != 0 else "")
		for i in range(j + 1, len(A)):
			append_data_to_table(*A[i], f[i], sum(A[i]) + f[i], s.pop(0) if j != 0 else "")
			s.append((sum(A[i]) + f[i]) - A[i][j] * (sum([x / A[j][j] for x in A[j]]) + f[j]/A[j][j]))
			koef = -A[i][j] / A[j][j]
			A[i] = list(np.array(A[j]) * koef + A[i])
			f[i] = f[j] * koef + f[i]
		append_data_to_table(*[x/A[j][j] for x in A[j]], f[j]/A[j][j], sum([
			x / A[j][j] for x in A[j]]) + f[j]/A[j][j], (sum(A[j]) + f[j])/A[j][j])
	s1 = sum(A[-1]) + f[-1]
	s2 = s.pop(0)
	append_data_to_table(*A[-1], f[-1], s1, s2)
	solutions = []
	control_solutions = []
	det = 1
	temp_arr = [A[i][i] for i in range(len(A))]
	for i in temp_arr:
		det *= i
	for i in range(len(A) - 1, -1, -1):
		temp = f[i]
		for j in range(1, len(A) - i):
			temp -= A[i][len(A) - j] * solutions[j - 1]
		temp /= A[i][i]
		solutions.append(temp)
	# first_solution = [x for x in solutions]
	sec_solutions = [x + 1 for x in solutions]
	ind = len(A) - 1
	for x in range(len(solutions)):
		append_data_to_table(*["1" if j == ind else "" for j in range(len(A))], solutions[x], sec_solutions[x], sec_solutions[x])
		ind -= 1
	solutions.reverse()
	return solutions, det


def octahedral_norm(matrix):
	return np.max([np.sum(np.abs(row)) for row in np.array(matrix)])


def cubic_norm(matrix):
	return octahedral_norm(np.array(matrix).T)


def cubic_r(vector):
	return max([abs(x) for x in vector])


def octahedral_r(vector):
	return sum([abs(x) for x in vector])


def spherical_r(vector):
	return sqrt(sum([x * x for x in vector]))

def find_inverse_matrix(A):
	A = list(A)
	bA = np.eye(len(A))
	for j in range(len(A) - 1):
		for i in range(j + 1, len(A)):
			koef = -A[i][j] / A[j][j]
			A[i] = list(np.array(A[j]) * koef + A[i])
			bA[i] = bA[j] * koef + bA[i]
	for i in range(len(A) - 1, -1, - 1):
		bA[i] = np.array(bA[i]) / A[i][i]
		A[i] = np.array(A[i]) / A[i][i]
		for j in range(i - 1, -1, -1):
			bA[j] = list(np.array(bA[j]) - np.array(bA[i]) * A[j][i])
			A[j] = list(np.array(A[j]) - np.array(A[i]) * A[j][i])
	return bA

if __name__ == "__main__":
	# find_solution(np.array([[0.14,0.24,-0.84], [1.07, -0.83, 0.56], [0.64, 0.43, -0.38]]),
	# 	np.array([1.11, 0.48,-0.83]))
	D = np.array([[6.22, 1.42, -1.72, 1.91],
	              [1.44, 5.33, 1.11, -1.82],
	              [-1.72, 1.11, 5.24, 1.42],
	              [1.91, -1.82, 1.42, 6.55]])
	k = 2
	C = np.eye(4)
	f = np.array([7.53, 6.06, 8.05, 8.06])
	solutions, det = find_solution(D + C * k, f)
	draw_table()
	r = (D + C * k).dot(np.array(solutions)) - f
	print(f"my x* is {solutions}")
	# print(f"correct x* is {np.linalg.solve(D + C * k, f)}")
	print(f"my det is {det}")
	print(f"correct det is {np.linalg.det(D + C * k)}")
	print(f"octahedral norm of matrix {octahedral_norm(D + C * k)}")
	print(f"cubic norm of matrix {cubic_norm(D + C * k)}")
	print(f"r is {r}")
	print(f"cubic norm of vector {cubic_r(r)}")
	print(f"octahedral norm of vector {octahedral_r(r)}")
	print(f"spherical norm of vector {spherical_r(r)}")