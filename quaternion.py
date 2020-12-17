import tensorflow as tf
from math import pi


    # def quat2format_scalar_last():
    #     return tf.transpose(tf.stack([x[:, 1], x[:, 2], x[:, 3], x[:, 0]]))
    #
    # def quat2format_scalar_first():
    #     return tf.transpose(tf.stack([x[:, -1], x[:, 0], x[:, 1], x[:, 2]]))

    # def distance(x, y):
    #     dot = tf.reduce_sum(x * y, -1)
    #     return tf.math.acos(tf.math.abs(dot))
# def omega_quat_left(x):
#     w = x[:, 0][:, None]
#     v = x[:, 1:]
#     n = x.shape[0]
#
#     omega00 = w
#     omega10 = -v
#     omega01 = v
#     omega11 = tf.eye(3, batch_shape=(n,))*w[...,None] + skew(v)
#     row0 = tf.concat([omega00, omega01], -1)[...,None]
#     row1 = tf.concat([omega10[:,None], omega11], 1)
#
#     return tf.concat([row0, row1], -1)
#
class Quaternion(object):
    """
    Implementation based on Riemannian manifold theory.
    """
    def __init__(self, q):
        self.q = q/(tf.linalg.norm(q, axis=-1)[...,None]+1e-15)
        self.qw = self.q[..., 0][:, None]
        self.qv = self.q[..., 1:]
        self.q_norm = tf.linalg.norm(self.q, axis=-1)[...,None]
        self.qv_norm = tf.linalg.norm(self.qv, axis=-1)[...,None]

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
            self._conjuguate = tf.concat([self.qw, -self.qv], -1)
        return self._conjuguate

    def conjuguate_q(self):
        return Quaternion(self.conjuguate)

    @property
    def u(self):
        if self._u is None:
            self._u = self.qv / (self.qv_norm + 1e-10)
        return self._u

    @property
    def theta(self):
        if self._theta is None:
            self._theta = 2*tf.math.atan2(self.qv_norm, self.qw)
        return self._theta

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            p1 = self.qw * other.qw - tf.reduce_sum(self.qv * other.qv, -1)[..., None]
            p2 = tf.linalg.cross(other.qv, self.qv) + self.qw * other.qv + other.qw * self.qv
            return tf.concat([p1, p2], -1)
        else:
            p1 = self.qw * other[:, 0][...,None] - tf.reduce_sum(self.qv * other[:, 1:], -1)[..., None]
            p2 = tf.linalg.cross(other[..., 1:], self.qv) + self.qw * other[..., 1:] + other[..., 0][...,None]* self.qv
            return tf.concat([p1, p2], -1)


    def exp_map(self, other):
        """
        This function maps a vector u lying on the tangent space of x0 into the manifold.
        :param other:
        :return:
        """
        norm_u = tf.math.sqrt(tf.reduce_sum(other * other, axis=-1))[...,None]
        x = self.q * tf.math.cos(norm_u) + other * tf.math.sin(norm_u) / norm_u
        return x

    def __sub__(self, other):
        """
        This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.
        :param other: a point on the manifold (sphere for quaternion)
        :return: vector in the tangent space of self
        """
        theta = self.distance(other)[...,None]
        u = (self.q - other.q * tf.math.cos(theta)) * theta/(tf.math.sin(theta)+1e-10)
        return u

    def distance(self, other):
        """
        This function computes the Riemannian distance between two points on the manifold.
        :param other: point on the manifold, Quaternion object
        :return: distance: manifold distance between x and y
        """
        dot = tf.reduce_sum(self.q * other.q , -1)
        dot = tf.clip_by_value(dot, -1+1e-4, 1-1e-4)
        # dot = tf.math.abs(dot)
        return tf.math.acos(dot)

    def axis_angle(self):
        return self.u, self.theta

    @property
    def rot_vector(self):
        return self.u*self.theta

    def log(self):
        return tf.concat([tf.math.log(self.q_norm), self.rot_vector()], -1)

    def exp(self):
        exp_q = tf.concat([tf.math.cos(0.5*self.qv_norm),
                           self.u * tf.math.sin(0.5*self.qv_norm)], -1)

        return exp_q*tf.math.exp(self.qw)

    def exp_q(self):
        return Quaternion(self.exp())

class Quaternion2(object):
    """
    Implementation based on Lie Group theory.
    """
    def __init__(self, q):
        self.q = q
        self.qw = self.q[:, 0][:, None]
        self.qv = self.q[:, 1:]
        self.q_norm = tf.linalg.norm(self.q, axis=-1)[:,None]
        self.qv_norm = tf.linalg.norm(self.qv, axis=-1)[:,None]

        self._inverse = None
        self._conjuguate = None
        self._u = None
        self._theta = None

    @property
    def inverse(self):
        if self._inverse is None:
            self._inverse = self.conjuguate/(self.q_norm+1e-15)
        return self._inverse

    @property
    def conjuguate(self):
        if self._conjuguate is None:
            self._conjuguate = tf.concat([self.qw, -self.qv], -1)
        return self._conjuguate

    def conjuguate_q(self):
        return Quaternion(self.conjuguate)

    @property
    def u(self):
        if self._u is None:
            self._u = self.qv / (self.qv_norm + 1e-15)
        return self._u

    @property
    def theta(self):
        if self._theta is None:
            self._theta = 2*tf.math.atan2(self.qv_norm, self.qw)
        return self._theta

    def __mul__(self, other):
        p1 = self.qw * other.qw - tf.reduce_sum(self.qv * other.qv, -1)[:, None]
        p2 = tf.linalg.cross(other.qv, self.qv) + self.qw * other.qv + other.qw * self.qv
        return tf.concat([p1, p2], -1)

    def __add__(self, other):
        return self * Quaternion(other.exp())

    def __sub__(self, other):
        return Quaternion(other.conjuguate_q() * self).log()

    def axis_angle(self):
        return self.u, self.theta

    def rot_vector(self):
        return self.u*self.theta

    def log(self):
        return tf.concat([tf.math.log(self.q_norm), self.rot_vector()], -1)

    def exp(self):
        exp_q = tf.concat([tf.math.cos(0.5*self.qv_norm),
                           self.u * tf.math.sin(0.5*self.qv_norm)], -1)

        return exp_q*tf.math.exp(self.qw)

    def exp_q(self):
        return Quaternion(self.exp())




