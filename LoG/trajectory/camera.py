import numpy as np
from .utils import getProjectionMatrix2,get_w2c_matrix




class Camera_view(object):
    def __init__(self, K, R, T, dist=None, name='', scale=1, znear=0.01, zfar=100.):
        self.K = K
        self.R = R
        self.T = T
        self.dist = dist
        self.name = name
        self.scale = scale
        self.znear = znear
        self.zfar = zfar

    def gene_K(self,Fovx=47,Fovy=65,x_size=2700,y_size=1800):
        fovxr=np.pi/180*Fovx
        fovyr=np.pi/180*Fovy
        fx=x_size/(2*np.tan(fovxr/2))
        fy=y_size/(2*np.tan(fovyr/2))
        cx=x_size/2
        cy=y_size/2
        self.K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    def feature_dict(self):
        feature_d={}
        feature_d['K']=self.K
        feature_d['R']=self.R
        feature_d['T']=self.T
        feature_d['dist']=self.dist
        feature_d['name']=self.name
        feature_d['scale']=self.scale
        feature_d['znear']=self.znear
        feature_d['zfar']=self.zfar
        feature_d['c2s_matrix']=self.c2s_matrix
        feature_d['w2c_matrix']=self.w2c_matrix
        feature_d['full_projection_matrix']=self.full_projection_matrix
        return feature_d




    @staticmethod
    def gene_from_coords(coords, K=None, dist=None, name_list=None, scale=1, znear=0.01, zfar=100.):
        """
        assume the fov 47 degree
        coord np list [N,3]
        K np list [N,4] fx fy cx cy
        """
        def gene_from_coord(coord,K,name,H=1800,W=2700):
            C = Camera_view(None, None, None, None, name, scale, znear, zfar)
            C.image_width=W
            C.image_height=H

            if (K == None):
                C.gene_K()
            else:
                C.K = K
            # C.R = np.eye(3)
            # C.T = coord / C.scale
            C.w2c_matrix=get_w2c_matrix(coord,np.array([0.,0.,0.]))
            C.R=C.w2c_matrix[:,:3]
            C.T=C.w2c_matrix[:,3]

            C.c2s_matrix=getProjectionMatrix2(C.K, C.image_height, C.image_width, znear, zfar)

            C.full_projection_matrix=C.w2c_matrix@C.c2s_matrix
            return C


        C_list=[]
        for i in range(len(coords)):
            k=K[i] if K is not None else None
            if name_list is not None:
                name = name_list[i]
            else:
                name=None
            C = gene_from_coord(coords[i],K,name)
            C_list.append(C)
        return C_list


'''
test
'''
if __name__ == '__main__':
    coords = np.array([[1, 1, 1],[2,2,2]])
    K = None
    dist = None
    name = None
    scale = 2
    znear = 0.01
    zfar = 100

    c = Camera_view.gene_from_coords(coords, K, dist, name, scale, znear, zfar)
    for camera in c :
        print(camera.feature_dict())
