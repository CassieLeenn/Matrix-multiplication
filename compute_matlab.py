#coding=utf-8
import matlab
import matlab.engine
import Generate_data as data
import numpy as np
import scipy.io as io

def Compute_in_matlab():
    A=data.matrix_A
    B=data.matrix_B
    A=np.array(A)
    B=np.array(B)

    save_path_A="./A.mat"
    save_path_B="./B.mat"
    io.savemat(save_path_A,{'A':A})
    io.savemat(save_path_B,{'B':B})

    eng=matlab.engine.start_matlab()

    t=eng.matrix_multiplication(nargout = 1) # 接受返回的一个参数
    print("matlab time:",t)
