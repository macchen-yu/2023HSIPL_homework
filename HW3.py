import numpy as np
import matplotlib.pyplot as plt

# Read Data
fp = r'./panel.npy'
data = np.load(fp, allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'), 'double')

p1 = him[7, 37, :]
p2 = him[20, 35, :]
p3 = him[34, 34, :]
p4 = him[47, 33, :]
p5 = him[59, 33, :]
p_list = [p1, p2, p3, p4, p5]


N = him.shape[0] * him.shape[1]  # 64*64 = 4096
# K
himK = np.reshape(him, (N, 169))
mean_K = np.mean(himK, 0)
#得到斜方差
K = np.dot((himK - mean_K).T, himK - mean_K) / N
inK = np.linalg.inv(K) #K inverse

# R
himR = np.reshape(him, (N, 169))
Rstdx = np.std(him[0, :, ])
Rstdy = np.std(him[:, 0, ])
mean_R = np.mean(himR, 0)
R = np.dot((himR - mean_R).T, himR - mean_R) / np.dot(np.dot(N, Rstdx), Rstdy)  # 169*4096  4096*169 =169*169
print(R.shape)
inR = np.linalg.inv(R) #R inverse



fig, axs = plt.subplots(4, 5, figsize=(15, 12))
# RXD
for i in range(N):
    rxd = np.zeros(N)
    for j in range(169):
        rxd[j] = (himK[i] - mean_K[j]).T @ inK @ (himK[i] - mean_K[j])
    rxd = np.reshape(rxd, (64, 64))
    axs[0, i].imshow(rxd, cmap='viridis',aspect='auto' )
    axs[0, i].set_title(f'RXD {i+1}',fontsize=6)
    plt.show()
# RMD 相關性 馬氏距離
    rmd = np.zeros(N)
    for j in range(N):
        rmd[j] = (p_list[i] - himR[j]).T @ inR @ (p_list[i] - himR[j])
    rmd = np.reshape(rmd, (64, 64))
    axs[1, i].imshow(rmd, cmap='viridis',aspect='auto')
    axs[1, i].set_title(f'RMD {i+1}',fontsize=6)

# CMFM 斜方差的匹配滤波器度量
    cmfm = (himK - mean_K) @ inK @(p_list[i] - mean_K).T
    cmfm = np.reshape(cmfm, (64, 64))
    axs[2, i].imshow(cmfm, cmap='viridis',aspect='auto')
    axs[2, i].set_title(f'CMFM {i+1}', fontsize=6)

# RMFM 相關性的匹配滤波器度量
    rmfm =  himK @ inR @ p_list[i]
    rmfm = np.reshape(rmfm, (64, 64))
    axs[3, i].imshow(rmfm, cmap='viridis',aspect='auto')
    axs[3, i].set_title(f'RMFM {i+1}', fontsize=6)

# plt.show()