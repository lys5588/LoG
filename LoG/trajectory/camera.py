import numpy as np
from .utils import getProjectionMatrix2,get_w2c_matrix,get_w2c_matrix_y_z,cal_rotate_matrix

import torch
import math

class ViewDataset():
    # @staticmethod
    # def init_camera(camera):
    #     width, height = camera['W'], camera['H']
    #     assert width != 0 and height != 0, f'width or height is 0: {width}, {height}'
    #     dist = camera['dist']
    #     if np.linalg.norm(dist) < 1e-5:
    #         mapx, mapy = None, None
    #         newK = camera['K'].copy()
    #     else:
    #         newK, roi = cv2.getOptimalNewCameraMatrix(camera['K'], camera['dist'],
    #                                                   (width, height), 0, (width, height), centerPrincipalPoint=True)
    #         mapx, mapy = cv2.initUndistortRectifyMap(camera['K'], camera['dist'], None, newK, (width, height), 5)
    #     return mapx, mapy, newK

    # def check_undis_camera(self, camname, cameras_cache, camera_undis, share_camera=False):
    #     if share_camera:
    #         cache_camname = 'cache'
    #     else:
    #         if '/' in camname:
    #             cache_camname = camname.split('/')[0]
    #         else:
    #             cache_camname = camname
    #
    #     if cache_camname not in cameras_cache:
    #         print(f'[{self.__class__.__name__}] init camera {cache_camname}')
    #         cameras_cache[cache_camname] = self.init_camera(camera_undis)
    #     mapx, mapy, newK = cameras_cache[cache_camname]
    #     camera = {
    #         'K': newK,
    #         'mapx': mapx,
    #         'mapy': mapy
    #     }
    #     for key in ['R', 'T', 'W', 'H', 'center']:
    #         camera[key] = camera_undis[key]
    #     return camera

    def __init__(self, views,device,**kwargs) -> None:
        # fxxk why put all param inthe default setting???
        super().__init__(**kwargs)
        self.views = views
        self.device=device




    def __len__(self):
        return len(self.views)


    def __getitem__(self, index):
        data = self.views[index]
        for key, val in data.items():
            if key in [
                "image_width",
                "image_height",
                "FoVy",
                "FoVx",
                "scale",
                "znear",
                "zfar",
                "name"
            ]:
                continue
            if isinstance(val, np.ndarray):
                data[key] = torch.FloatTensor(data[key]).to(self.device)
            elif torch.is_tensor(data[key]):
                data[key] = data[key].float().to(self.device)
            else:
                import ipdb
                ipdb.set_trace()

        ret = {
            'index': index,
            'camera': data,
        }

        return ret


class Camera_view(object):
    def __init__(self, center,K, R, T, name='', scale=1, znear=0.01, zfar=100.):
        self.world_center=center
        self.K = K
        self.R = R
        self.T = T
        self.name = name
        self.scale = scale
        self.znear = znear
        self.zfar = zfar

    def gene_K(self,Fovx=47,Fovy=65,x_size=2700,y_size=1800):
        self.image_height=y_size
        self.image_width= x_size
        self.fovxr=np.pi/180*Fovx
        self.fovyr=np.pi/180*Fovy
        fx=x_size/(2*np.tan(self.fovxr/2))
        fy=y_size/(2*np.tan(self.fovyr/2))
        cx=x_size/2
        cy=y_size/2
        self.K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

    def feature_dict(self):
        feature_d={}
        feature_d['camera_center']=self.world_center
        feature_d['K']=self.K
        feature_d['R']=self.R
        feature_d['T']=self.T
        feature_d['FoVx']=self.fovxr
        feature_d['FoVy']=self.fovyr
        feature_d['image_width']=self.image_width
        feature_d['image_height']=self.image_height
        feature_d['name']=self.name
        feature_d['scale']=self.scale
        feature_d['znear']=self.znear
        feature_d['zfar']=self.zfar
        feature_d['w2c_matrix']=self.w2c_matrix
        feature_d['world_view_transform']=self.w2c_matrix
        feature_d['full_proj_transform']=self.full_projection_matrix
        feature_d['full_projection_matrix']=self.full_projection_matrix
        return feature_d


    def gen_R_by_axix(self,rot_type,rot_theta,R_origin):
        '''
        coordinate:
        x to right
        y to down
        z to inside

        '''

        #first generate the rotation matrix to world coordinate
        if  R_origin is not None:
            self.rot_2w=R_origin.T
        else:
            a_2w = np.array([1., 0., 0.])
            theta_2w = math.radians(-45)
            self.rot_2w = cal_rotate_matrix(a_2w, theta_2w)

        rot_w2=np.eye(3)
        if rot_type == "z":
            a = np.array([0, 0, 1])
            theta = math.radians(rot_theta)
            rot_w2 =  cal_rotate_matrix(a, theta)
        elif rot_type == 'x':
            a = np.array([1, 0, 0])
            theta = math.radians(rot_theta)
            rot_w2 = cal_rotate_matrix(a, theta)
        elif rot_type == 'y':
            a = np.array([0, 1, 0])
            theta = math.radians(rot_theta)
            rot_w2 = cal_rotate_matrix(a, theta)
        elif rot_type== 'y_world':
            a = np.array([0, 1, 0])
            theta = math.radians(rot_theta)
            rot_w2 = self.rot_2w.T @ cal_rotate_matrix(a, theta)
        else:
            pass
        pass
        self.rot_w2=rot_w2
        self.R=rot_w2@self.rot_2w

    def gen_R_towards(self,p_target,type,rot_theta):
        '''
        生成一个朝向目标位置的视角
        1. 成像平面与p_camera 和 p_target 为平行
        2. 成像平面与xoy平面平行


        '''
        a_2w = np.array([1., 0., 0.])
        theta_2w = math.radians(-45)
        self.rot_2w = cal_rotate_matrix(a_2w, theta_2w)


        z_axis=p_target-self.world_center
        z_axis=z_axis/np.linalg.norm(z_axis)

        z_up=np.array([0.,0.,1.])
        x_axis=np.cross(z_axis,z_up)
        x_axis/=np.linalg.norm(x_axis)

        y_axis=np.cross(z_axis,x_axis)
        y_axis/=np.linalg.norm(y_axis)

        rot_w2 = np.array([x_axis,y_axis,z_axis])
        self.rot_w2 = rot_w2
        # self.R = rot_w2
        self.R = rot_w2 @ self.rot_2w









    @staticmethod
    def gene_from_coords(coords, K=None, dist=None, name_list=None, scale=1, znear=0.01, zfar=100.,rot_type="x-z45",rot_theta=45,R_origin=None,coord_origin=None):
        """
        assume the fov 47 degree
        coord np list [N,3]
        K np list [N,4] fx fy cx cy
        """
        def gene_from_coord(coord,K,name,H=1800,W=2700):
            '''
            adapt scale in the future
            '''


            C = Camera_view(coord,None,  None, None, name, scale, znear, zfar)
            C.image_width=W
            C.image_height=H

            #intrinsic feature
            if (K == None):
                C.gene_K()
            else:
                C.K = K
            C.c2s_matrix = getProjectionMatrix2(C.K, C.image_height, C.image_width, znear, zfar)
            # C.R = np.eye(3)
            # C.T = coord / C.scale
            #extrinsic feature
            C.w2c_matrix=np.eye(4)

            if rot_type=="toward":
                p_target=np.array([0.,0.,0.])
                C.gen_R_towards(p_target,type,rot_theta)
            else:
                # C.R=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
                C.world_center=C.world_center-coord_origin
                C.gen_R_by_axix(rot_type, rot_theta,R_origin)

            C.w2c_matrix[:3,:3]=C.R
            C.T=- C.R @ coord
            C.w2c_matrix[:3,3]=C.T


            C.full_projection_matrix=C.w2c_matrix@C.c2s_matrix
            return C

        def gene_from_coord_w2(coord,K,name,H=1800,W=2700):
            '''
            adapt scale in the future
            逻辑如下：
            center coordinate in world
            R @ P_world + T= P_camera
            '''


            C = Camera_view(coord,None,  None, None, name, scale, znear, zfar)
            C.image_width=W
            C.image_height=H

            #intrinsic feature
            if (K == None):
                C.gene_K()
            else:
                C.K = K
            C.c2s_matrix = getProjectionMatrix2(C.K, C.image_height, C.image_width, znear, zfar)
            # C.R = np.eye(3)
            # C.T = coord / C.scale

            #extrinsic feature
            C.w2c_matrix=np.eye(4)
            C.gen_R_by_axix(type,rot_theta)
            C.w2c_matrix[:3,:3]=C.R.T
            C.T=- C.R @ coord
            C.w2c_matrix[:3,3]=C.T


            C.full_projection_matrix=C.w2c_matrix@C.c2s_matrix
            return C

        C_list=[]
        for i in range(len(coords)):
            k=K[i] if K is not None else None
            if name_list is not None:
                name = name_list[i]
            else:
                name=""
            C = gene_from_coord(coords[i],K,name)
            C_list.append(C)
        return C_list


'''
test
'''
if __name__ == '__main__':
    # from utils import getProjectionMatrix2, get_w2c_matrix, get_w2c_matrix_y_z, cal_rotate_matrix
    # coords = np.array([[1, 1, 1],[2,2,2]])
    # K = None
    # dist = None
    # name = None
    # scale = 2
    # znear = 0.01
    # zfar = 100
    #
    # c = Camera_view.gene_from_coords(coords, K, dist, name, scale, znear, zfar)
    # for camera in c :
    #     print(camera.feature_dict())
    # view_center=np.array([-4.600131734467356,0.6818177352577112,-0.7217177952270529])
    view_center = np.array([1., 1., 1.])
    cam= Camera_view.gene_from_coords(view_center.reshape((-1, 3)), type="toward",rot_theta=45)

    p_cam=cam[0].R @ cam[0].world_center + cam[0].T

    p_zero=cam[0].R @ np.array([0.,0.,0.]) + cam[0].T
    print(cam)