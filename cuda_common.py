#!/usr/bin/env python
# -*- coding:utf-8 -*-


from numba import cuda, float32
import numba
import numpy as np
import math
import time
import Generate_data as data

TPB = 16


@numba.jit(nopython=True)  # numba就是通过JIT加速了python代码。
def matmul_cpu(A, B, C):
    for y in range(B.shape[1]):
        for x in range(A.shape[0]):
            tmp = 0.
            for k in range(A.shape[1]):
                tmp += A[x, k] * B[k, y]
            C[x, y] = tmp


@cuda.jit()
def matmul_gpu(A, B, C):  # 当前grid中线程的位置
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit()
def matmul_shared_mem(A, B, C):
    sA = cuda.shared.array((TPB, TPB), dtype=float32)
    sB = cuda.shared.array((TPB, TPB), dtype=float32)
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    if x >= C.shape[0] and y >= C.shape[1]:
        return
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]
        cuda.syncthreads()
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]
        cuda.syncthreads()
    C[x, y] = tmp



def Common():
    print("Ordinary matrix multiplication")
    matrix_A = data.matrix_A
    matrix_B = data.matrix_B

    # print("numpy:",np.matmul(matrix_A,matrix_B))

    C_cpu = np.full((matrix_A.shape[0], matrix_B.shape[1]), 0, np.float)

    # start in GPU
    A_global_mem = cuda.to_device(matrix_A)
    B_global_mem = cuda.to_device(matrix_B)

    C_global_mem = cuda.device_array((matrix_A.shape[0], matrix_B.shape[1]))

    threads_per_block = (TPB, TPB)
    blocks_per_grid_x = int(math.ceil(matrix_A.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(math.ceil(matrix_B.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    print("start processing in GPU")
    start_gpu = time.time()
    matmul_gpu[blocks_per_grid, threads_per_block](A_global_mem, B_global_mem, C_global_mem)
    cuda.synchronize()
    end_gpu = time.time()
    time_gpu = end_gpu - start_gpu
    print("GPU time(Global memory):" + str(time_gpu))

    # C_global_gpu = C_global_mem.copy_to_host()
    # print("common:",C_global_gpu)

