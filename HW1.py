import numpy as np
import matplotlib.pyplot  as plt
from pathlib import Path 

def SAM(vector1:np.ndarray, vector2:np.ndarray):
    angle_radians = np.arccos(np.clip(vector1@vector2/ (np.linalg.norm(vector1) * np.linalg.norm(vector2)), -1.0, 1.0))
    return angle_radians
   
def SID(x:np.ndarray,y:np.ndarray):
    p = (x / np.sum(x))
    q = (y / np.sum(y))
    return np.sum(p * np.log(p / q) + q * np.log(q / p))
def SID_TAN(x:np.ndarray,y:np.ndarray):
    p = (x / np.sum(x))
    q = (y / np.sum(y))
    Sid_value = np.sum(p * np.log(p / q) + q * np.log(q / p))
    Sam_value =  SAM(x,y)
    return Sid_value*np.tan(Sam_value)
def SID_sin(x:np.ndarray,y:np.ndarray):
    p = (x / np.sum(x))
    q = (y / np.sum(y))
    Sid_value = np.sum(p * np.log(p / q) + q * np.log(q / p))
    Sam_value =  SAM(x,y)
    return Sid_value*np.sin(Sam_value)
# def sam
# def ED():

def get_five_purepixel_in_arry(path=r'panel.npy')-> np.ndarray:
    data = np.load(path,allow_pickle=True)
    item = data.item()
    groundtruth = np.array(item.get('groundtruth'), 'double')
    him = np.array(item.get('HIM'),'double')
    plt.figure
    p1=him[7,37,:]
    p2=him[20,35,:]
    p3=him[34,34,:]
    p4=him[47,33,:]
    p5=him[59,33,:]
    p_all_refle= np.vstack((p1,p2,p3,p4,p5))
    return p_all_refle

def print_SAM_value():
    p_all_refle=get_five_purepixel_in_arry()
    print("=========SAM結果=============================")
    for i in range(5):  
        for j in range(5): 
            if i==j:
                continue
            result=SAM(p_all_refle[i,:],p_all_refle[j,:])
            result=round(result, 6)
            print(f'pi:{i+1}, pj:{j+1},SAM結果: {result}')
def print_SID_value():
    p_all_refle=get_five_purepixel_in_arry()
    print("=========SID結果=============================")
    for i in range(5):  
        for j in range(5):  
            if i==j:
                continue
            result=SID(p_all_refle[i,:],p_all_refle[j,:])
            result=round(result, 6)
            print(f'pi:{i+1}, pj:{j+1},SID結果: {result}')
def print_SID_TAN_value():
    p_all_refle=get_five_purepixel_in_arry()
    print("=========SID_TAN結果==========================")
    for i in range(5):  
        for j in range(5): 
            if i==j:
                continue
            result=SID_TAN(p_all_refle[i,:],p_all_refle[j,:])
            result=round(result, 6)
            print(f'pi:{i+1}, pj:{j+1},SID_TAN結果: {result}')
def print_SID_sin_value():
    p_all_refle=get_five_purepixel_in_arry()
    print("=========SID_sin結果=============================")
    for i in range(5):  
        for j in range(5):  
            if i==j:
                continue
            result=SID_sin(p_all_refle[i,:],p_all_refle[j,:])
            result=round(result, 6)
            print(f'pi:{i+1}, pj:{j+1},SID_sin結果: {result}')
def plot():
    p_all_refle=get_five_purepixel_in_arry()
    for i in range(5):
        plt.plot(p_all_refle[i,:],label=f'p{i+1}')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    print_SAM_value()
    print_SID_value()
    print_SID_TAN_value()
    print_SID_sin_value()
    # plot()
