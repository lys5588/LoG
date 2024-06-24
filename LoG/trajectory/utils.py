import numpy as np
import math
def getProjectionMatrix2(K, H, W, znear, zfar):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]

    P = np.zeros((4, 4), dtype=np.float32)

    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign

    return P


def get_w2c_matrix(O1,O2,R1=np.array([0,0,1])):
    '''
    默认观察视角高于xoy平面
    '''
    z_axis=O2-O1
    z_axis/=math.sqrt(np.dot(z_axis,z_axis.T))
    x_axis=np.cross(np.array([0,0,1]),z_axis)
    x_axis/=math.sqrt(np.dot(x_axis,x_axis.T))
    y_axis=np.cross(x_axis,z_axis)
    y_axis/=math.sqrt(np.dot(y_axis,y_axis.T))
    R=np.vstack((x_axis,y_axis,z_axis))
    T=np.dot(R,O1-O2)
    w2c_matrix=np.hstack((R,T.reshape(3,1)))

    return w2c_matrix

if __name__=="__main__":
    O1=np.array([0.,0.,0.])
    O2=np.array([1.,1.,1.])
    w2c_matrix=get_w2c_matrix(O1,O2)
    print(w2c_matrix)