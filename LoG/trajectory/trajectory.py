import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class Trajectory(object):
    """
    points: [N,3]
    """
    def __init__(self,points):
        self.init_points=points

    def set_num_points(self,num_point):
        self.num_points=num_point

    def interpolation_path(self,strategy="slerp"):
        pointlines=[]
        if strategy=="slerp":
            for i in range(len(self.init_points)):
                st_ed_pos=[self.init_points[i],self.init_points[i+1]]
                interp_points=self.interpolate_path_slerp(st_ed_pos)

            pointlines+=self.init_points[-1]
        return pointlines
    def interpolate_path_slerp(self,st_ed_pos):
        """
        -working on it
        interpolate the trajectory using slerp
        """
        st=st_ed_pos[0]
        ed=st_ed_pos[1]
        key_rots= R.from_matrix()