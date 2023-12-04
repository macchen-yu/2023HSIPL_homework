import numpy as np
import matplotlib.pyplot  as plt
 
fp = r'panel.npy'
data = np.load(fp,allow_pickle=True)
item = data.item()
groundtruth = np.array(item.get('groundtruth'), 'double')
him = np.array(item.get('HIM'),'double')
p1=him[7,37,:]
p2=him[20,35,:]
p3=him[34,34,:]
p4=him[47,33,:]
p5=him[59,33,:]
plt.plot(p1, label='p1')
plt.plot(p2, label='p2')
plt.plot(p3, label='p3')
plt.plot(p4, label='p4')
plt.plot(p5, label='p5')
plt.legend()
plt.show()

