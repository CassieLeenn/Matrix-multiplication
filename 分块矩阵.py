import Generate_data as data
from numba import cuda, float32
import time
import numpy as np
import cuda_strassen
import math
from numba import jit
import numba


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
    elif row >= x_middle and col < y_middle and row <rows:
        A21[row - x_middle, col] = A[row, col]
    # 右下部分
    elif row >= x_middle and col >= y_middle and row <rows and col < columns:
        A22[row - x_middle, col - y_middle] = A[row, col]
    # 右上部分
    elif row < x_middle and col >= y_middle and col<columns:
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


#CPU function

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

    # M1=(A11+A22)(B11+B22)
    A1122 = cuda.device_array((A11.shape[0], A11.shape[1]))
    B1122 = cuda.device_array((B11.shape[0], B11.shape[1]))
    M1 = cuda.device_array((A1122.shape[0], B1122.shape[1]))

    # M2=(A21+A22)B11
    A2122 = cuda.device_array((A21.shape[0], A21.shape[1]))
    M2 = cuda.device_array((A2122.shape[0], B11.shape[1]))

    # M3=A11(B12-B22)
    B12_22 = cuda.device_array((B12.shape[0], B12.shape[1]))
    M3 = cuda.device_array((A11.shape[0], B12_22.shape[1]))

    # M4 = A22(B21-B11)
    B21_11 = cuda.device_array((B21.shape[0], B21.shape[1]))
    M4 = cuda.device_array((A22.shape[0], B21_11.shape[1]))

    # M5 = (A11+A12)B22
    A1112 = cuda.device_array((A11.shape[0], A11.shape[1]))
    M5 = cuda.device_array((A1122.shape[0], B22.shape[1]))

    # M6 =(A21-A11)(B11+B12)
    A21_11 = cuda.device_array((A21.shape[0], A21.shape[1]))
    B1112 = cuda.device_array((B11.shape[0], B11.shape[1]))
    M6 = cuda.device_array((A21_11.shape[0], B1112.shape[1]))

    # M7= (A12-A22)(B21+B22)
    A12_22 = cuda.device_array((A12.shape[0], A12.shape[1]))
    B2122 = cuda.device_array((B21.shape[0], B21.shape[1]))
    M7 = cuda.device_array((A12_22.shape[0], B2122.shape[1]))

    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    C11 = cuda.device_array((M1.shape[0], M1.shape[1]))
    C12 = cuda.device_array((M3.shape[0], M5.shape[1]))
    C21 = cuda.device_array((M2.shape[0], M4.shape[0]))
    C22 = cuda.device_array((M1.shape[0], M1.shape[1]))

    M14 = cuda.device_array((M1.shape[0], M1.shape[1]))
    M15 = cuda.device_array((M1.shape[0], M1.shape[1]))

    M12 = cuda.device_array((M1.shape[0], M1.shape[1]))
    M13 = cuda.device_array((M1.shape[0], M1.shape[1]))

    # M1=(A11+A22)(B11+B22)
    kernel_add[blocks_per_grid, threads_per_block](A11, A22, A1122)
    kernel_add[blocks_per_grid, threads_per_block](B11, B22, B1122)
    # M2=(A21+A22)B11
    kernel_add[blocks_per_grid, threads_per_block](A21, A22, A2122)
    # M3=A11(B12-B22)
    kernel_sub[blocks_per_grid, threads_per_block](B12, B22, B12_22)
    # M4 = A22(B21-B11)
    kernel_sub[blocks_per_grid, threads_per_block](B21, B11, B21_11)
    # M5 = (A11+A12)B22
    kernel_add[blocks_per_grid, threads_per_block](A11, A12, A1112)
    # M6 =(A21-A11)(B11+B12)
    kernel_sub[blocks_per_grid, threads_per_block](A21, A11, A21_11)
    kernel_add[blocks_per_grid, threads_per_block](B11, B12, B1112)
    # M7= (A12-A22)(B21+B22)
    kernel_sub[blocks_per_grid, threads_per_block](A12, A22, A12_22)
    kernel_add[blocks_per_grid, threads_per_block](B21, B22, B2122)
    cuda.synchronize()

    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A1122, B1122, M1)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A1122, B1122, M1)

    # M2=(A21+A22)B11
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A2122, B11, M2)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A2122, B11, M2)

    # M3=A11(B12-B22)
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A11, B12_22, M3)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A11, B12_22, M3)

    # M4 = A22(B21-B11)
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A22, B21_11, M4)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A22, B21_11, M4)

    # M5 = (A11+A12)B22
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A1112, B22, M5)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A1112, B22, M5)

    # M6 =(A21-A11)(B11+B12)
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A21_11, B1112, M6)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A21_11, B1112, M6)

    # M7= (A12-A22)(B21+B22)
    if is_share:
        kernel_share_matmul[blocks_per_grid, threads_per_block](A12_22, B2122, M7)
    else:
        kernel_matmul[blocks_per_grid, threads_per_block](A12_22, B2122, M7)

    cuda.synchronize()
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # C11 = M1+M4-M5+M7
    kernel_add[blocks_per_grid, threads_per_block](M1, M4, M14)
    kernel_sub[blocks_per_grid, threads_per_block](M14, M5, M15)
    kernel_add[blocks_per_grid, threads_per_block](M15, M7, C11)

    # C12 = M3+M5
    kernel_add[blocks_per_grid, threads_per_block](M3, M5, C12)

    # C21 = M2+M4
    kernel_add[blocks_per_grid, threads_per_block](M2, M4, C21)

    # C22 = M1-M2+M3+M6
    kernel_sub[blocks_per_grid, threads_per_block](M1, M2, M12)
    kernel_add[blocks_per_grid, threads_per_block](M12, M3, M13)
    kernel_add[blocks_per_grid, threads_per_block](M13, M6, C22)

    # C = cuda.device_array((A.shape[0], B.shape[1]))
    # kernel_merge[blocks_per_grid, threads_per_block](C11, C12, C21, C22, C)
    # cuda.synchronize()
    # print(C.copy_to_host())

    C_row1 = np.hstack((C11, C12))
    C_row2 = np.hstack((C21, C22))

    C = np.vstack((C_row1, C_row2))
    # print("numpy hstack", C)

    return C



@cuda.jit()
def kernel_divide2(A,num,P,C):
    '''

    :param A:  要分块的矩阵
    :param num: 多少个矩阵
    :param P: 每个矩阵的维度
    :param C: 存储分块后的各矩阵
    :return:
    '''


    row,col=cuda.grid(2)

    if row< num and col <num:   # 一个线程处理一个
        for i in range(P):
            for j in range(P) :
                C[row,col,i,j]=A[int(row*P)+i,int(col*P)+j]



def kernel_block_multiplication(A,B,blocks_per_grid, threads_per_block,C):
    '''

    :param A: 被分块后的矩阵A [num,num,2^k,2^k]
    :param B: 被分块后的矩阵B [num,num,2^k,2^k]
    :param blocks_per_grid: cuda 初始化
    :param threads_per_block: cuda 初始化
    :param temp : 临时矩阵
    :return:
    '''
    # row,col=cuda.grid(2)
    num=A.shape[0]
    P=A.shape[2]


    # if row <num and col < num:
        # 每一个分块利用Strassen
        # 返回值后拼接回去 temp [2^k,2^k]
    for row in range(num):
        for col in range(num):

    # if row<num and col < num:
            for t in range(num):
                if t==0:
                    # 返回的是cuda 的类型
                    temp=matrix_multiplication(
                        A[row,t],B[t,col],blocks_per_grid,threads_per_block,False)
                    # inter=temp.copy_to_host()
                    inter=temp   #私有的还是公有的？
                    # print(inter.shape)
                else :
                    temp1=matrix_multiplication(
                        A[row, t], B[t, col], blocks_per_grid, threads_per_block, False)
                    # inter2=temp1.copy_to_host()
                    inter=inter+temp1

            # print("inter:",inter)


            for i in range(inter.shape[0]):
                for j in range(inter.shape[1]):
                    C[row*P+i,col*P+j]=inter[i,j]
            # print(C)
    return C


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


if __name__ == '__main__':

    matrix_A=data.matrix_A
    matrix_B=data.matrix_B

    C_numpy=np.matmul(matrix_A,matrix_B)

    print(np.matmul(matrix_A,matrix_B))

    print("\n")

    M = N = matrix_A.shape[0]

    TPB = 16
    flag =0
    if M % 2==1:
        M=M+1
        N=N+1

        P=N

        flag=1

        # initialization grid (另一个核函数)
        threads_per_block = (TPB, TPB)  # thread 在 block 中 16*16
        blocks_per_grid_x = int(math.ceil(P / threads_per_block[0]))  # ceil是返回大于的数
        blocks_per_grid_y = int(math.ceil(P / threads_per_block[1]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)  # grid 里面 有 的block

        A = cuda.device_array((P, P))
        B = cuda.device_array((P, P))
        kernel_marix_expand[blocks_per_grid, threads_per_block](matrix_A, A)
        kernel_marix_expand[blocks_per_grid, threads_per_block](matrix_B, B)
    else:
        A=matrix_A
        B=matrix_B
    K=0

    while M%2==0:
        K=K+1
        M=int(M/2)

    num=int(N/pow(2,K))   # num * num 个 2^K
    A_divide=cuda.device_array((num,num,pow(2,K),pow(2,K)))     # 四维
    B_divide = cuda.device_array((num, num, pow(2, K), pow(2, K)))

    P=pow(2,K)  # Strassen 矩阵相乘就是这个维度了

    # 加速分块部分
    # initialization grid (另一个核函数)
    threads_per_block = (TPB, TPB)  # thread 在 block 中 16*16
    blocks_per_grid_x = int(math.ceil(P / threads_per_block[0]))  # ceil是返回大于的数
    blocks_per_grid_y = int(math.ceil(P / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)  # grid 里面 有 的block
    kernel_divide2[blocks_per_grid,threads_per_block](A,num,P,A_divide)
    kernel_divide2[blocks_per_grid, threads_per_block](B, num, P, B_divide)

    # 利用矩阵相乘部分
    # C=cuda.device_array((A.shape[0],A.shape[1]))
    C=np.zeros((A.shape[0],A.shape[1]))
    temp=cuda.device_array((pow(2,K),pow(2,K)))

    threads_per_block_mul = (TPB, TPB)  # thread 在 block 中 16*16
    blocks_per_grid_x_mul = int(math.ceil(P / threads_per_block_mul[0]))  # ceil是返回大于的数
    blocks_per_grid_y_mul = int(math.ceil(P / threads_per_block_mul[1]))
    blocks_per_grid_mul = (blocks_per_grid_x_mul, blocks_per_grid_y_mul)  # grid 里面 有 的block

    start_time= time.time()
    C=kernel_block_multiplication(
        A_divide,B_divide,blocks_per_grid_mul, threads_per_block_mul,C)
    end_time=time.time()


    final_result = cuda.device_array((matrix_A.shape[0], matrix_B.shape[1]))  # 最终计算出来的结果

    if flag==1:
        kernel_matrix_shrink[blocks_per_grid, threads_per_block](C, matrix_A, final_result)
        final_result = final_result.copy_to_host()  # 转换成numpy类型
        # print("final result:",final_result)
    # else:
        # print(C)


    print("time:" ,end_time-start_time)









