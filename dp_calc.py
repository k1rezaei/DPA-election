import numpy as np
import json

def find_array(K = 1210):
    A = np.zeros((K, K,), dtype=int)

    for i in range(K):
        for j in range(K):
            if i == 0 and j == 0:
                A[i, j] = 0
            elif i <= 1 and j <= 2:
                A[i, j] = 1
            elif i <= 2 and j <= 1:
                A[i, j] = 1
            elif j <= 1:
                A[i, j] = (i + 1) // 2
            elif i <= 1:
                A[i, j] = (j + 1) // 2
            else:
                A[i, j] = 1 + min(A[i - 2, j - 1], A[i - 1, j - 2])
    return A
    
def get_value(A: np.ndarray, g1:int, g2:int):
    return A[g1, g2]

def save_json(B:np.ndarray):
    with open('data/array_v2.json', 'w') as f:
        f.write(json.dumps(B.tolist()))
    return 

def load_json():
    with open('data/array_v2.json', 'r') as f:
        return np.array(json.loads(f.read()), dtype=int)

def debug(B, g1, g2):
    print(f'g1, g2: {g1, g2} --> {B[g1][g2]}')

if __name__ == '__main__':
    K = 1210
    A = find_array(K)
    B = np.zeros((K - 1, K - 1, ), dtype=int)

    for i in range(K-1):
        for j in range(K - 1):
            B[i, j] = get_value(A, i, j)

    save_json(B)

    B = load_json()
    debug(B, 0, 0)
    debug(B, 0, 1)
    debug(B, 10, 10)
    debug(B, 2, 2)
    debug(B, 3, 3)
    debug(B, 0, 10)
    debug(B, 10, 10)
    debug(B, 6, 10)
    debug(B, 7, 10)

    

