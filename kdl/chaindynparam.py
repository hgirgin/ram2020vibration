from .utils import *
from .joint import JointType
from .frame import Twist, dot
from .rbi import Rotation, ArticulatedBodyInertia
import numpy as np



class ChainDynParam(object):
    def __init__(self, chain, g=None):
        self.chain = chain

        self.nb_joints = self.chain.nb_joint
        self.nb_segments = self.chain.nb_segm
        self.g = 9.81 if g is None else g
        self.Ic = [None]*self.nb_segments # list of tk.ArticulatedRigidBody
        self.X = [None]*self.nb_segments # list of tk.Frame
        self.S = [None]*self.nb_segments # list of tk.Twist
        # self.H = tf.Variable(tf.zeros((self.nb_joints, self.nb_joints)))$
        self.H_ = [[None] * self.nb_joints] * self.nb_joints

    def inertia_matrix(self, q):
        k = 0
        batch_size = q.shape[0]
        # self.H = tf.Variable(tf.zeros((batch_size, self.nb_joints, self.nb_joints)))
        self.H_ = [[tf.zeros_like(q[:,0]) for i in range(self.nb_joints)] for j in range(self.nb_joints)]

        ## Sweep from root to leaf
        for i in range(self.nb_segments):
            segment_i = self.chain.segments[i]

            ## Collect RigidBodyInertia
            self.Ic[i] = segment_i.I.make_batch(batch_size=batch_size)
            if segment_i.joint.type is not JointType.NoneT: ## in cpp it says fixed, is it the same?
                q_ = q[:, k]
                k +=1
            else:
                q_ = tf.zeros(batch_size)
            self.X[i] = segment_i.pose(q_) # type: Frame
            self.S[i] = self.X[i].M.inv(segment_i.twist(q_, 1.0)) # type: Twist

        ## Sweep from leaf to root
        k = self.nb_joints-1

        for i in range(self.nb_segments-1, -1, -1):
            segment_i = self.chain.segments[i]
            if i!= 0:
                # pass
                ## assumption that previous segment is parent
                self.Ic[i - 1] = self.Ic[i - 1] + self.Ic[i] * self.X[i]
            F = self.Ic[i]*self.S[i]

            if segment_i.joint.type is not JointType.NoneT: ## in cpp it says fixed, is it the same?
                H_kk = dot(self.S[i], F) #+ segment_i.joint.inertia   # add joint inertia
                # self.H[:,k,k].assign(H_kk)
                self.H_[k][k] = H_kk

                j=k # counter variable for the joints
                l=i # counter variable for the segments

                while l!=0: # go from leaf to root starting at i
                    #assumption that previous segment is parent
                    F = self.X[l]*F #calculate the unit force (cfr S) for every segment: F[l-1]=X[l]*F[l]
                    l += -1 # go down a segment
                    if self.chain.segments[l].joint.type is not JointType.NoneT:
                        j += -1
                        H_kj = dot(F, self.S[l])
                        # self.H[:,k,j].assign(H_kj) # here you actually match a certain not fixed joint with a segment
                        # self.H[:,j,k].assign(H_kj)
                        self.H_[k][j] = H_kj
                        self.H_[j][k] = H_kj

                k += -1 #this if-loop should be repeated nb_joint times (k=nb_joint-1 to k=0)
        # return self.H
        return tf.transpose(tf.stack(self.H_), (2,0,1))

def joint_list_to_kdl(q):
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = PyKDL.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl

