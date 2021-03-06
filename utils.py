from scipy.spatial.transform import Rotation as R

import pybullet as p
import numpy as np
import time
import pyscreenshot as ImageGrab
import pinocchio as pin
import transforms3d

def rotvec2mat(x):
    return R.as_matrix(R.from_rotvec(x))


def save_screenshot(x,y,w,h,file_name, to_show='False'):
    # part of the screen
    im=ImageGrab.grab(bbox=(x,y,w,h))
    if to_show:
        im.show()
    # save to file
    im.save(file_name)
    return im


def normalize(x):
    return x/np.linalg.norm(x)
        
def set_q(robot_id, joint_indices, q, set_base = False):
    if set_base:
        localInertiaPos = np.array(p.getDynamicsInfo(robot_id,-1)[3])
        q_root = q[0:7]
        ori = q_root[3:]
        Rbase = np.array(p.getMatrixFromQuaternion(ori)).reshape(3,3)
        shift_base = Rbase.dot(localInertiaPos)
        pos = q_root[:3]+shift_base
        p.resetBasePositionAndOrientation(robot_id,pos,ori)
        q_joint = q[7:]
    else:
        q_joint = q
    
    #set joint angles
    for i in range(len(q_joint)):
        p.resetJointState(robot_id, joint_indices[i], q_joint[i])


def vis_traj(qs, vis_func, dt=0.1):
    for q in qs:
        vis_func(q)
        time.sleep(dt)


def get_joint_limits(robot_id, indices):
    lower_limits = []
    upper_limits = []
    for i in indices:
        info = p.getJointInfo(robot_id, i)
        lower_limits += [info[8]]
        upper_limits += [info[9]]
    limits = np.vstack([lower_limits, upper_limits])
    return limits

    
def check_joint_limits(q, joint_limits):
    """
    Return True if within the limit
    """
    upper_check = False in ((q-joint_limits[0]) > 0)
    lower_check = False in ((joint_limits[1]-q) > 0)
    if upper_check or lower_check:
        return False
    else:
        return True
    
def calc_dist_limit(q, joint_limits):
    lower_error = joint_limits[0]-q
    lower_check = (lower_error > 0)
    lower_error = lower_error*lower_check
    upper_error = q-joint_limits[1]
    upper_check = (upper_error > 0)
    upper_error = upper_error*upper_check
    error = lower_error-upper_error
    return error
    
def euler2quat(rpy, axes='sxyz'):
    #euler sxyz: used by Manu's codes
    return rectify_quat(transforms3d.euler.euler2quat(*rpy, axes=axes))

def rectify_quat(quat):
    #transform from transforms3d format (w,xyz) to pybullet and pinocchio (xyz, w)
    quat_new = np.concatenate([quat[1:], quat[0:1]])
    return quat_new

def w2quat(q):
    angle = np.linalg.norm(q)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = normalize(q), np.linalg.norm(q)
    w = p.getQuaternionFromAxisAngle(ax, angle)
    return np.array(w)

def w2mat(w):
    angle = np.linalg.norm(w)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = w/angle, angle
    R = pin.AngleAxis.toRotationMatrix(pin.AngleAxis(angle, ax))
    return R

def w2quat(q):
    angle = np.linalg.norm(q)
    if abs(angle) < 1e-7:
        ax = np.array([1,0,0])
    else:
        ax, angle = normalize(q), np.linalg.norm(q)
    w = p.getQuaternionFromAxisAngle(ax, angle)
    return np.array(w)

def quat2w(q):
    ax, angle = p.getAxisAngleFromQuaternion(q)
    return np.array(ax)*angle

def get_link_base(robot_id, frame_id):
    '''
    Obtain the coordinate of the link frame, according to the convention of pinocchio (at the link origin, 
    instead of at the COM as in pybullet)
    '''
    p1 = np.array(p.getLinkState(robot_id,frame_id)[0])
    ori1 = np.array(p.getLinkState(robot_id,frame_id)[1])
    R1 = np.array(p.getMatrixFromQuaternion(ori1)).reshape(3,3)
    p2 = np.array(p.getLinkState(robot_id,frame_id)[2])
    return  p1 - R1.dot(p2), ori1

    
def create_primitives(shapeType=2, rgbaColor=[1, 1, 0, 1], pos = [0, 0, 0], radius = 1, length = 2, halfExtents = [0.5, 0.5, 0.5], baseMass=1, basePosition = [0,0,0]):
    visualShapeId = p.createVisualShape(shapeType=shapeType, rgbaColor=rgbaColor, visualFramePosition=pos, radius=radius, length=length, halfExtents = halfExtents)
    collisionShapeId = p.createCollisionShape(shapeType=shapeType, collisionFramePosition=pos, radius=radius, height=length, halfExtents = halfExtents)
    bodyId = p.createMultiBody(baseMass=baseMass,
                      baseInertialFramePosition=[0, 0, 0],
                      baseVisualShapeIndex=visualShapeId,
                      baseCollisionShapeIndex=collisionShapeId,    
                      basePosition=basePosition,
                      useMaximalCoordinates=True)
    return visualShapeId, collisionShapeId, bodyId

    
#### Code to modify concave objects in pybullet
#name_in =  rl.datapath + '/urdf/bookcase_old.obj'
#name_out = rl.datapath + '/urdf/bookcase.obj'
#name_log = "log.txt"
#p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=10000000 )

def Rot_z(angle):
    w = np.array([0,0,angle])
    Rz = w2mat(w)
    return Rz
