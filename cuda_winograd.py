'''
Coppersmith-Winograd算法 with cuda
'''
from functools import lru_cache
import numpy as np
import time
import numba
import math
from numba import cuda, float32
import Generate_data as data
import pandas as pd

TPB = 16


@cuda.jit
def kernel_divide(A, A11, A12, A21, A22):
    # 返回四个子矩阵：分别为矩阵的左上，右上，左下，右下
    # 设计思路: 如果分配到的该线程满足在区间范围内，则放入
    rows = A.shape[0]
    columns = A.shape[1]
    x_middle = rows // 2  # // 表示先做除法，然后向下取整
    y_middle = columns // 2
    row, col = cuda.grid(2)

    # 左上部分
    if row < x_middle and col < y_middle:
        A11[row, col] = A[row, col]
    # 左下部分
    elif row >= x_middle and col < y_middle and row < rows:
        A21[row - x_middle, col] = A[row, col]
    # 右下部分
    elif row >= x_middle and col >= y_middle and row < rows and col < columns:
        A22[row - x_middle, col - y_middle] = A[row, col]
    # 右上部分
    elif row < x_middle and col >= y_middle and col < columns:
        A12[row, col - y_middle] = A[row, col]


@cuda.jit
def kernel_merge(C11, C12, C21, C22, C):
    # 基本思路: 在某一范围则可以进行放进去。
    row, col = cuda.grid(2)
    x_middle = C11.shape[0]
    y_middle = C11.shape[1]

    if row < x_middle and col < y_middle:
        C[row, col] = C11[row, col]
    elif row < x_middle and col >= y_middle:
        C[row, col] = C12[row, col - y_middle]
    elif row >= x_middle and col < y_middle:
        C[row, col] = C21[row - x_middle, col]
    elif row >= x_middle and col >= y_middle:
        C[row, col] = C22[row - x_middle, col - y_middle]


@cuda.jit
def kernel_add(A, B, C):  # 矩阵相加 可以直接理解为每个线程处理一个加法
    '''
    矩阵相加核函数

    :param A: 第一个矩阵
    :param B: 第二个矩阵
    :param C: 记录矩阵相加的结果
    :return:
    '''

    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] + B[row, col]


@cuda.jit
def kernel_sub(A, B, C):
    '''
    矩阵相减核函数

    :param A: 第一个矩阵
    :param B: 第二个矩阵
    :param C: 记录矩阵相减的结果
    :return:
    '''

    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = A[row, col] - B[row, col]


# 这是共享内存的加速方法
@cuda.jit()
def kernel_share_matmul(A, B, C):
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


@cuda.jit()
def kernel_matmul(A, B, C):  # 每个线程来处理求一个C矩阵中的元素值，最终并行执行得到结果矩阵。
    '''
    矩阵相乘核函数

    :param A: 第一个矩阵
    :param B: 第二个矩阵
    :param C: 记录矩阵相乘的结果
    :return:
    '''
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit()
def kernel_marix_expand(A, C):  # 增加到2的n次方维度为止
    '''

    :param A: 要扩展的矩阵
    :param C: 扩展后的矩阵
    :return:
    '''
    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:
        C[row, col] = A[row, col]
    else:
        C[row, col] = 0


@cuda.jit()
def kernel_matrix_shrink(C, A, final_result):
    '''

    :param C: 扩展后的矩阵
    :param A: 原来的矩阵
    :param final_result: 最终变回原来维度的矩阵
    :return:
    '''

    row, col = cuda.grid(2)
    if row < A.shape[0] and col < A.shape[1]:  # A只是作为边
        final_result[row, col] = C[row, col]


def matrix_multiplication(A, B, blocks_per_grid, threads_per_block, is_share):
    '''
    Copper_Winograd 算法实现

    :param A: 第一个矩阵
    :param B: 第二个矩阵
    :param is_share: 是否采用全局加速
    :return: 矩阵相乘结果
    '''
    row, col = A.shape

    new_row = int(row / 2)
    new_col = int(col / 2)

    rows = A.shape[0]
    columns = A.shape[1]
    x_middle = int(rows / 2)

    y_middle = int(columns / 2)

    A11 = cuda.device_array((x_middle, y_middle))
    A12 = cuda.device_array((x_middle, y_middle))
    A21 = cuda.device_array((x_middle, y_middle))
    A22 = cuda.device_array((x_middle, y_middle))

    B11 = cuda.device_array((x_middle, y_middle))
    B12 = cuda.device_array((x_middle, y_middle))
    B21 = cuda.device_array((x_middle, y_middle))
    B22 = cuda.device_array((x_middle, y_middle))

    kernel_divide[blocks_per_grid, threads_per_block](A, A11, A12, A21, A22)
    kernel_divide[blocks_per_grid, threads_per_block](B, B11, B12, B21, B22)

    # S1=A21+A22
    # S2=S1-A11
    # S3=A11-A21
    # S4=A12-S2
    S1 = cuda.device_array((A21.shape[0], A21.shape[1]))
    S2 = cuda.device_array((S1.shape[0], A11.shape[1]))
    S3 = cuda.device_array((A11.shape[0], A21.shape[1]))
    S4 = cuda.device_array((A12.shape[0], S2.shape[1]))

    # S1=A21+A22
    kernel_add[blocks_per_grid, threads_per_block](A21, A22, S1)
    # S2=S1-A11
    kernel_sub[blocks_per_grid, threads_per_block](S1, A11, S2)
    # S3=A11-A21
    kernel_sub[blocks_per_grid, threads_per_block](A11, A21, S3)
    # S4=A12-S2
    kernel_sub[blocks_per_grid, threads_per_block](A12, S2, S4)

    # T1=B12-B11
    # T2=B22-T1
    # T3=B22-B12
    # T4=T2-B21
    T1 = cuda.device_array((B12.shape[0], B11.shape[1]))
    T2 = cuda.device_array((B22.shape[0], T1.shape[1]))
    T3 = cuda.device_array((B22.shape[0], B12.shape[1]))
    T4 = cuda.device_array((T2.shape[0], B21.shape[1]))

    # T1=B12-B11
    kernel_sub[blocks_per_grid, threads_per_block](B12, B11, T1)
    # T2=B22-T1
    kernel_sub[blocks_per_grid, threads_per_block](B22, T1, T2)
    # T3=B22-B12
    kernel_sub[blocks_per_grid, threads_per_block](B22, B12, T3)
    # T4=T2-B21
    kernel_sub[blocks_per_grid, threads_per_block](T2, B21, T4)
    # cuda.syncthreads()

    # M1 = A11B11
    # M2 = A12B21
    # M3 = S4B22
    # M4 = A22T4
    # M5 = S1T1
    # M6 = S2T2
    # M7 = S3T3
    M1 = cuda.device_array((A11.shape[0], B11.shape[1]))
    M2 = cuda.device_array((A12.shape[0], B21.shape[1]))
    M3 = cuda.device_array((S4.shape[0], B22.shape[1]))
    M4 = cuda.device_array((A22.shape[0], T4.shape[1]))
    M5 = cuda.device_array((S1.shape[0], T1.shape[1]))
    M6 = cuda.device_array((S2.shape[0], T2.shape[1]))
    M7 = cuda.device_array((S3.shape[0], T3.shape[1]))

    if is_share is False:
        # M1 = A11B11
        kernel_matmul[blocks_per_grid, threads_per_block](A11, B11, M1)
        # M2 = A12B21
        kernel_matmul[blocks_per_grid, threads_per_block](A12, B21, M2)
        # M3 = S4B22
        kernel_matmul[blocks_per_grid, threads_per_block](S4, B22, M3)
        # M4 = A22T4
        kernel_matmul[blocks_per_grid, threads_per_block](A22, T4, M4)
        # M5 = S1T1
        kernel_matmul[blocks_per_grid, threads_per_block](S1, T1, M5)
        # M6 = S2T2
        kernel_matmul[blocks_per_grid, threads_per_block](S2, T2, M6)
        # M7 = S3T3
        kernel_matmul[blocks_per_grid, threads_per_block](S3, T3, M7)
    else:
        # M1 = A11B11
        kernel_share_matmul[blocks_per_grid, threads_per_block](A11, B11, M1)
        # M2 = A12B21
        kernel_share_matmul[blocks_per_grid, threads_per_block](A12, B21, M2)
        # M3 = S4B22
        kernel_share_matmul[blocks_per_grid, threads_per_block](S4, B22, M3)
        # M4 = A22T4
        kernel_share_matmul[blocks_per_grid, threads_per_block](A22, T4, M4)
        # M5 = S1T1
        kernel_share_matmul[blocks_per_grid, threads_per_block](S1, T1, M5)
        # M6 = S2T2
        kernel_share_matmul[blocks_per_grid, threads_per_block](S2, T2, M6)
        # M7 = S3T3
        kernel_share_matmul[blocks_per_grid, threads_per_block](S3, T3, M7)
    # cuda.syncthreads()

    # U1 =M1 +M2
    # U2=M1+M6
    # U3=U2+M7
    # U4=U2+M5
    # U5=U4+M3
    # U6=U3-M4
    # U7=U3+M5

    U1 = cuda.device_array((M1.shape[0], M2.shape[1]))
    U2 = cuda.device_array((M1.shape[0], M6.shape[1]))
    U3 = cuda.device_array((U2.shape[0], M7.shape[1]))
    U4 = cuda.device_array((U2.shape[0], M5.shape[1]))
    U5 = cuda.device_array((U4.shape[0], M3.shape[1]))
    U6 = cuda.device_array((U3.shape[0], M4.shape[1]))
    U7 = cuda.device_array((U3.shape[0], M5.shape[1]))

    # U1 =M1 +M2
    kernel_add[blocks_per_grid, threads_per_block](M1, M2, U1)
    # U2=M1+M6
    kernel_add[blocks_per_grid, threads_per_block](M1, M6, U2)
    # U3=U2+M7
    kernel_add[blocks_per_grid, threads_per_block](U2, M7, U3)
    # U4=U2+M5
    kernel_add[blocks_per_grid, threads_per_block](U2, M5, U4)
    # U5=U4+M3
    kernel_add[blocks_per_grid, threads_per_block](U4, M3, U5)
    # U6=U3-M4
    kernel_sub[blocks_per_grid, threads_per_block](U3, M4, U6)
    # U7=U3+M5
    kernel_add[blocks_per_grid, threads_per_block](U3, M5, U7)
    # cuda.syncthreads()

    C11 = U1
    C12 = U5
    C21 = U6
    C22 = U7

    C_row1 = np.hstack((C11, C12))
    C_row2 = np.hstack((C21, C22))

    C = np.vstack((C_row1, C_row2))
    # print("numpy hstack", C)

    # C = cuda.device_array((A.shape[0], B.shape[1]))
    # kernel_merge[blocks_per_grid, threads_per_block](U1, U5, U6, U7, C)

    return C


def Winograd():
    print("Coppersmith-Winograd")
    # 随机生成的矩阵
    matrix_A = data.matrix_A
    matrix_B = data.matrix_B

    # print("numpy:",np.matmul(matrix_A,matrix_B))
    flag = 0

    # if (math.log(matrix_A.shape[0], 2) - np.floor(math.log(matrix_A.shape[0], 2))):  # 如果不是2的次幂维度
    if matrix_A.shape[0] % 2 == 1:

        flag = 1
        matrix_A = cuda.to_device(matrix_A)
        matrix_B = cuda.to_device(matrix_B)

        N = matrix_A.shape[0]
        # M = math.ceil(math.log(N, 2))
        # P = int(pow(2, M))

        P = N + 1

        col = np.zeros((P - 1, 1))
        row = np.zeros((1, P))

        A = np.c_[matrix_A, col]
        A = np.r_[A, row]

        B = np.c_[matrix_B, col]
        B = np.r_[B, row]

        A = cuda.to_device(A)
        B = cuda.to_device(B)
        # initialization grid (另一个核函数)
        # threads_per_block = (TPB, TPB)  # thread 在 block 中 16*16
        # blocks_per_grid_x = int(math.ceil(P / threads_per_block[0]))  # ceil是返回大于的数
        # blocks_per_grid_y = int(math.ceil(P / threads_per_block[1]))
        # blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)  # grid 里面 有 的block
        #
        # A = cuda.device_array((P, P))
        # B = cuda.device_array((P, P))
        # kernel_marix_expand[blocks_per_grid, threads_per_block](matrix_A, A)
        # kernel_marix_expand[blocks_per_grid, threads_per_block](matrix_B, B)

    else:
        A = matrix_A
        B = matrix_B

    C_global_mem = cuda.device_array((A.shape[0], B.shape[1]))

    # initialization grid
    threads_per_block = (TPB, TPB)  # thread 在 block 中 16*16
    blocks_per_grid_x = int(math.ceil(A.shape[0] / threads_per_block[0]))  # ceil是返回大于的数
    blocks_per_grid_y = int(math.ceil(B.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)  # grid 里面 有 的block

    # CopperWingoard()算法执行

    # print("start in share GPU")
    # start_time = time.time()
    # C_global_mem = matrix_multiplication(A, B, blocks_per_grid, threads_per_block, True)
    # # C_global_mem = C_global_mem.copy_to_host()  # 转换成numpy类型
    # end_time = time.time()

    print("start in global GPU")
    start_time_1 = time.time()
    C_global_mem = matrix_multiplication(A, B, blocks_per_grid, threads_per_block, False)
    # C_global_mem = C_global_mem.copy_to_host()  # 转换成numpy类型
    end_time_1 = time.time()

    # print("Winograd:",C_global_mem)

    final_result = cuda.device_array((matrix_A.shape[0], matrix_B.shape[1]))  # 最终计算出来的结果

    if flag == 1:
        kernel_matrix_shrink[blocks_per_grid, threads_per_block](C_global_mem, matrix_A, final_result)
        final_result = final_result.copy_to_host()  # 转换成numpy类型
        result = pd.DataFrame(final_result)
        # result.to_csv("./winogard_share_"+str(matrix_A.shape[0])+"_result.csv")
        # print("final_result:",final_result)
    else:
        result = pd.DataFrame(C_global_mem)
        # result.to_csv("./winogard_global_" + str(matrix_A.shape[0]) + "_result.csv")
        # print("Winograd:",C_global_mem)

    # print("GPU time spend in share:", end_time - start_time)
    print("GPU time spend in global:", end_time_1 - start_time_1)
