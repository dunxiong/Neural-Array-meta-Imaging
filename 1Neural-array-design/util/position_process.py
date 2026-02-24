import scipy.io as sio
import numpy as np

def position_degui(y1,y2,merge_position_threshold):

    y2_temp1 = y1[0,0]
    y2_temp2 = y1[1,0]
    number = 1
    P = np.array([0])
    for i in range(1,y1.shape[1]):
        temp = np.square(y1[0,0]-y1[0,i]) + np.square(y1[1,0]-y1[1,i])
        temp = np.sqrt(temp)
        if temp <= merge_position_threshold:
            y2_temp1 = y2_temp1 + y1[0,i]
            y2_temp2 = y2_temp2 + y1[1,i]
            number = number + 1
            P_temp = np.array([i])
            P = np.concatenate((P,P_temp))
            
    y1_1 = np.delete(y1[0,:],P)
    y1_1 = y1_1[np.newaxis,:]
    y1_2 = np.delete(y1[1,:],P)
    y1_2 = y1_2[np.newaxis,:]
    y1 = np.concatenate((y1_1,y1_2),axis = 0)
    
    y2_temp1 = y2_temp1 / number
    y2_temp2 = y2_temp2 / number
    y2_temp = np.array([[y2_temp1],[y2_temp2]])
    y2 = np.concatenate((y2,y2_temp),axis = 1)
    return y1,y2

def position_degui_cubic(y1,y2,xita1,xita2,merge_position_threshold):

    y2_temp1 = y1[0,0]
    y2_temp2 = y1[1,0]
    xita2_temp = xita1[0]
    
    number = 1
    P = np.array([0])
    for i in range(1,y1.shape[1]):
        temp = np.square(y1[0,0]-y1[0,i]) + np.square(y1[1,0]-y1[1,i])
        temp = np.sqrt(temp)
        if temp <= merge_position_threshold:
            y2_temp1 = y2_temp1 + y1[0,i]
            y2_temp2 = y2_temp2 + y1[1,i]
            xita2_temp = xita2_temp + xita1[i]
            number = number + 1
            P_temp = np.array([i])
            P = np.concatenate((P,P_temp))
            
    y1_1 = np.delete(y1[0,:],P)
    y1_1 = y1_1[np.newaxis,:]
    y1_2 = np.delete(y1[1,:],P)
    y1_2 = y1_2[np.newaxis,:]
    y1 = np.concatenate((y1_1,y1_2),axis = 0)

    xita11 = np.delete(xita1,P)
    
    y2_temp1 = y2_temp1 / number
    y2_temp2 = y2_temp2 / number
    y2_temp = np.array([[y2_temp1],[y2_temp2]])
    y2 = np.concatenate((y2,y2_temp),axis = 1)

    xita2_temp1 = xita2_temp / number
    xita2_temp = np.array([xita2_temp1])
    xita2 = np.concatenate((xita2,xita2_temp),axis = 0)
    
    return y1,y2,xita11,xita2


def save_new_position_map(input_name,out_name,x_position, y_position,merge_position_threshold):


    array_position_x = x_position
    array_position_y = y_position

    array_position_x = array_position_x[np.newaxis,:]
    array_position_y = array_position_y[np.newaxis,:]

    position = np.concatenate((array_position_x,array_position_y),axis=0)

    range_temp = sio.loadmat(input_name)
    range_volume_value = range_temp['Range'].astype(np.float32)
    sio.savemat('before_process.mat', {'position':position,'Range':range_volume_value})

    y1 = position
    y2 = np.array([[0],[0]])

    for i in range(position.shape[1]):
        if y1.shape[1] == 0:
            break
        y1,y2 = position_degui(y1,y2,merge_position_threshold)
    position_new = y2[:,1:]
    print(position_new.shape)

    [x,y] = np.mgrid[0.5:1600:1,0.5:1600:1]
    x = x * 1.725e-6
    y = y * 1.725e-6
    range_new = np.zeros([1600*1600,9])
    X = x.reshape(-1)
    Y = y.reshape(-1)
    print(X.shape)

    M = 1600*100
    N = np.int32(1600*1600/M)

    for i in range(N):
        [Rxo,Rx] = np.meshgrid(position_new[0,:],X[M*(i):M*(i+1)])
        [Ryo,Ry] = np.meshgrid(position_new[1,:],Y[M*(i):M*(i+1)])
        print(Ryo.shape)
        R = np.square(Rx-Rxo) + np.square(Ry-Ryo)      
        I = np.argsort(R,axis=1,kind='mergesort')
        range_new[M*(i):M*(i+1),:] = I[:,0:9]
        print(I.shape)

    Range_new = np.reshape(range_new,[1600,1600,9])
    Range_new = np.transpose(Range_new, (1, 0, 2))   # 需要转至
            
    sio.savemat(out_name, {'position':position_new,'Range':Range_new})

def save_new_position_map_cubic(input_name,out_name,x_position, y_position,xita, merge_position_threshold):

    array_position_x = x_position
    array_position_y = y_position
    xita = xita

    array_position_x = array_position_x[np.newaxis,:]
    array_position_y = array_position_y[np.newaxis,:]

    position = np.concatenate((array_position_x,array_position_y),axis=0)

    range_temp = sio.loadmat(input_name)
    range_volume_value = range_temp['Range'].astype(np.float32)
    arfa = range_temp['arfa'].astype(np.float32)
    
    sio.savemat('before_process.mat',
                {'position': position, 'Range': range_volume_value, 'xita_offset': xita, 'arfa': arfa})

    y1 = position
    y2 = np.array([[0],[0]])

    xita1 = xita
    xita2 = np.array([0])

    for i in range(position.shape[1]):
        if y1.shape[1] == 0:
            break
        y1, y2, xita1, xita2 = position_degui_cubic(y1, y2, xita1, xita2, merge_position_threshold)
    position_new = y2[:, 1:]
    xita_new = xita2[1:]

    print(position_new.shape)
    print(xita_new.shape)

    [x,y] = np.mgrid[0.5:1600:1,0.5:1600:1]
    x = x * 1.725e-6
    y = y * 1.725e-6
    range_new = np.zeros([1600*1600,9])
    X = x.reshape(-1)
    Y = y.reshape(-1)
    print(X.shape)

    M = 1600*100
    N = np.int32(1600*1600/M)

    for i in range(N):
        [Rxo,Rx] = np.meshgrid(position_new[0,:],X[M*(i):M*(i+1)])
        [Ryo,Ry] = np.meshgrid(position_new[1,:],Y[M*(i):M*(i+1)])
        print(Ryo.shape)
        R = np.square(Rx-Rxo) + np.square(Ry-Ryo)      
        I = np.argsort(R,axis=1,kind='mergesort')
        range_new[M*(i):M*(i+1),:] = I[:,0:9]
        print(I.shape)

    Range_new = np.reshape(range_new,[1600,1600,9])
    Range_new = np.transpose(Range_new, (1, 0, 2))   # 需要转至
            
    sio.savemat(out_name, {'position': position_new, 'Range': Range_new, 'xita_offset': xita_new, 'arfa': arfa })
