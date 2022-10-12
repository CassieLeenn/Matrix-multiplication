import numpy as np
import scipy.io as io
#同时进行数据预处理
print("Generate data now ...")
m_seed=10
np.random.seed(m_seed)

# 矩阵维度
N=128
print("维度:",N)
matrix_A=np.random.rand(N,N)
matrix_B=np.random.rand(N,N)

save_path_A="./A.mat"
save_path_B="./B.mat"
io.savemat(save_path_A,{'A':matrix_A})
io.savemat(save_path_B,{'B':matrix_B})






