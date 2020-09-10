from scipy.spatial.transform import Rotation as R

def rotvec2mat(x):
    return R.as_matrix(R.from_rotvec(x))
