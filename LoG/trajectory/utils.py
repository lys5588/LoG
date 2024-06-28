import numpy as np
import math
from scipy.spatial.transform import Rotation as Rot
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


def get_w2c_matrix(O1,R1=np.array([0,0,1])):
    '''
    默认观察视角高于xoy平面
    O1,O2分别为两个坐标系在世界坐标系原点的位置
    按照colmap设定
    R@P_c+T=P_w
    '''
    z_axis=-O1
    z_axis=z_axis/np.linalg.norm(z_axis)

    x_axis=np.cross(np.array([0,0,1]),z_axis)
    x_axis=x_axis/np.linalg.norm(x_axis)

    y_axis=np.cross(x_axis,z_axis)
    y_axis=y_axis/np.linalg.norm(y_axis)

    R=np.vstack(np.array([x_axis,y_axis,z_axis])).T
    chect=np.dot(R,R.T)
    T=np.dot(-R.T,O1)
    w2c_matrix=np.hstack((R,T.reshape(3,1)))

    return w2c_matrix

def get_w2c_matrix_y_z(O1,degree=45):
    '''
    生成一个从这个坐标向y轴正方向与z轴负方向成45°的视角
    '''
    dir_vec=np.array([0.,1.,0.])
    up_vec = np.array([0, 0, -1])

    alpha=math.radians(degree)
    x_axis=np.array([-1,0,0])
    z_axix=np.array([0,1.*math.cos(alpha),-1.*math.sin(alpha)])
    y_axis=np.cross(x_axis,z_axix)

    R=np.vstack(np.array([x_axis,y_axis,z_axix])).T
    T = np.dot(-R.T, O1)
    w2c_matrix=np.hstack((R,T.reshape(3,1)))
    return w2c_matrix

def cal_rotate_matrix(a,theta):
    """
    a:旋转轴
    theta:旋转角度
    这里生成的旋转矩阵对应规则为 R@P_camera=P_world
    """
    mrp=a*np.tan(theta/4)
    rotation_matrix=Rot.from_mrp(mrp).as_matrix()
    return rotation_matrix




if __name__=="__main__":
    # O1=np.array([0.,0.,0.])
    # w2c_matrix=get_w2c_matrix_y_z(O1)
    # R=w2c_matrix[:,:3]
    # T=w2c_matrix[:,3]
    # c1=np.dot(R.T,np.array([1.,1.,0.]))+T
    # c2=np.dot(R.T,np.array([0.,0.,0.]))+T
    # c3=np.dot(R.T,np.array([2.,0.,0.]))+T
    # print(w2c_matrix)

    P1=np.array([0.70710678 ,0.70710678 ,0.         ,1.        ])
    a=np.array([1.,0.,0.])
    theta=math.radians(90)
    R=cal_rotate_matrix(a,theta)
    print(R)
    print(np.dot(R.T,P1))

