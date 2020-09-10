import numpy as np
from math import pi

class QuaternionNP(object):
    """
    Implementation based on Riemannian manifold theory.
    """
    def __init__(self, q):
        self.q = q/(np.linalg.norm(q, axis=-1)[:,None]+1e-15)
        self.qw = self.q[:, 0][:, None]
        self.qv = self.q[:, 1:]
        self.q_norm = np.linalg.norm(self.q, axis=-1)[:,None]
        self.qv_norm = np.linalg.norm(self.qv, axis=-1)[:,None]

        self._inverse = None
        self._conjuguate = None
        self._u = None
        self._theta = None

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = self.conjuguate/(self.q_norm+1e-10)
        return self._inverse

    @property
    def conjuguate(self):
        if self._conjuguate is None:
            self._conjuguate = np.concatenate([self.qw, -self.qv], -1)
        return self._conjuguate

    def conjuguate_q(self):
        return QuaternionNP(self.conjuguate)

    @property
    def u(self):
        if self._u is None:
            self._u = self.qv / (self.qv_norm + 1e-10)
        return self._u

    @property
    def theta(self):
        if self._theta is None:
            self._theta = 2*np.atan2(self.qv_norm, self.qw)
        return self._theta

    def __mul__(self, other):
        p1 = self.qw * other.qw - np.sum(self.qv * other.qv, -1)[:, None]
        p2 = np.cross(other.qv, self.qv) + self.qw * other.qv + other.qw * self.qv
        return np.concatenate([p1, p2], -1)

    def exp_map(self, other):
        """
        This function maps a vector u lying on the tangent space of x0 into the manifold.
        :param other:
        :return:
        """
        norm_u = (np.sqrt(np.sum(other * other, axis=-1))[:,None]+1e-15)
        x = self.q * np.cos(norm_u) + other * np.sin(norm_u) / norm_u
        return x

    def __sub__(self, other):
        """
        This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
        :param other: a point on the manifold (sphere for quaternion)
        :return: vector in the tangent space of self
        """
        theta = self.distance(other)[:,None]
        u = (self.q - other.q * np.cos(theta)) * theta/(np.sin(theta)+1e-10)
        return u

    def distance(self, other):
        """
        This function computes the Riemannian distance between two points on the manifold.
        :param other: point on the manifold, Quaternion object
        :return: distance: manifold distance between x and y
        """
        dot = np.sum(self.q * other.q, -1)
        dot = np.clip(dot, -1+1e-4, 1-1e-4)
        dot = np.abs(dot)
        return np.arccos(dot)

    def axis_angle(self):
        return self.u, self.theta

    @property
    def rot_vector(self):
        return self.u*self.theta

    def log(self):
        return np.concatenate([np.log(self.q_norm), self.rot_vector()], -1)

    def exp(self):
        exp_q = np.concatenate([np.cos(0.5*self.qv_norm),
                           self.u * np.sin(0.5*self.qv_norm)], -1)

        return exp_q*np.exp(self.qw)

    def exp_q(self):
        return QuaternionNP(self.exp())

# class Quaternion2NP(object):
#     """
#     Implementation based on Lie Group theory.
#     """
#     def __init__(self, q):
#         self.q = q
#         self.qw = self.q[:, 0][:, None]
#         self.qv = self.q[:, 1:]
#         self.q_norm = tf.linalg.norm(self.q, axis=-1)[:,None]
#         self.qv_norm = tf.linalg.norm(self.qv, axis=-1)[:,None]
#
#         self._inverse = None
#         self._conjuguate = None
#         self._u = None
#         self._theta = None
#
#     @property
#     def inverse(self):
#         if self._inverse is None:
#             self._inverse = self.conjuguate/(self.q_norm+1e-15)
#         return self._inverse
#
#     @property
#     def conjuguate(self):
#         if self._conjuguate is None:
#             self._conjuguate = tf.concat([self.qw, -self.qv], -1)
#         return self._conjuguate
#
#     def conjuguate_q(self):
#         return Quaternion(self.conjuguate)
#
#     @property
#     def u(self):
#         if self._u is None:
#             self._u = self.qv / (self.qv_norm + 1e-15)
#         return self._u
#
#     @property
#     def theta(self):
#         if self._theta is None:
#             self._theta = 2*tf.math.atan2(self.qv_norm, self.qw)
#         return self._theta
#
#     def __mul__(self, other):
#         p1 = self.qw * other.qw - tf.reduce_sum(self.qv * other.qv, -1)[:, None]
#         p2 = tf.linalg.cross(other.qv, self.qv) + self.qw * other.qv + other.qw * self.qv
#         return tf.concat([p1, p2], -1)
#
#     def __add__(self, other):
#         return self * Quaternion2NP(other.exp())
#
#     def __sub__(self, other):
#         return Quaternion2NP(other.conjuguate_q() * self).log()
#
#     def axis_angle(self):
#         return self.u, self.theta
#
#     def rot_vector(self):
#         return self.u*self.theta
#
#     def log(self):
#         return tf.concat([tf.math.log(self.q_norm), self.rot_vector()], -1)
#
#     def exp(self):
#         exp_q = tf.concat([tf.math.cos(0.5*self.qv_norm),
#                            self.u * tf.math.sin(0.5*self.qv_norm)], -1)
#
#         return exp_q*tf.math.exp(self.qw)
#
#     def exp_q(self):
#         return Quaternion2NP(self.exp())
#
#
#
#
