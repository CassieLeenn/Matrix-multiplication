import matplotlib.pyplot as plt
import numpy as np

matlab_A=np.array([0.0082749,0.418465,0.449943,0.801593,1.249154,2.32085,3.796887,5.64703,7.977474,11.17054])
cuda_common=np.array([4.259449,7.524277,13.59398,25.95095,26.75592,32.39994,32.65549,38.17806,39.21003,39.45575])

matlab_mi=np.array([26.1745,32.8400,46.0924,72.4100,80.3987,128.6076])
cuda_strassen=np.array([34.72776,47.19222,60.40099,68.80843,62.69964,78.13814])
cuda_wingoard=np.array([46.93838,66.52089,93.45823,92.19235,108.9583,126.0219])

cpu_cannon=np.array([99.46,120.35,143.23,168.09,194.95,223.79])
x=np.arange(1000,11000,1000)

common=np.array([14.155,60.19,236.638,11004.752])
strassen=np.array([7.9932,40.7892,75.045,2657.756])
matlab=np.array([0.082794,0.418465,0.449943,11.17054])

# plt.plot([10000,11000,12000,13000,14000,15000],matlab_mi,label="matlab")
# plt.plot([10000,11000,12000,13000,14000,15000],cuda_strassen,label="cuda_strassen")
# plt.plot([10000,11000,12000,13000,14000,15000],cuda_wingoard,label="cuda_winograd")
# plt.plot([10000,11000,12000,13000,14000,15000],cpu_cannon,label="cpu_cannon")
plt.plot([1000,2000,3000,10000],matlab,label="matlab")
plt.plot([1000,2000,3000,10000],strassen,label="strassen")
plt.plot([1000,2000,3000,10000],common,label="common")


# plt.plot(x,matlab_A,label="matlab")
# plt.plot(x,cuda_common,label="cuda_common")
plt.title("Matrix multiplication")
plt.xlabel("Dimension")
plt.ylabel("time(s)")
plt.legend()
plt.show()