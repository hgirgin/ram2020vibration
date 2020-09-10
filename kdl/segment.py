from .rbi import Rotation
from .joint import Joint
from .utils import *


class Segment(object):
    def __init__(self, joint, f_tip, I=None, child_name='', link=None):
        """
		Segment of a kinematic chain
        :param I:
        :type I: tk.RigidBodyInertia
		:param joint:
		:type joint: tk.Joint
		:param f_tip:
		:type f_tip: tk.Frame
		"""
        self.joint = joint
        self.f_tip = joint.pose(0.).inv() * f_tip
        self.child_name = child_name
        self.I = I
        self.link = link

    def __eq__(self, oher):
        pass

    def pose(self, q):
        """

        :param q:
        :return: tk.Frame
        """
        return self.joint.pose(q) * self.f_tip

    def twist(self, q, qdot=0.):
        """

        :param q:
        :param qdot:
        :return: tk.Twist
        """
        return self.joint.twist(qdot).ref_point(Rotation(self.joint.pose(q).m)*self.f_tip.p)

    def set_frame_to_tip(self, f_tip_new):
        self.f_tip = self.joint.pose(0).inv()*f_tip_new

