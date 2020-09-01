from .utils.tf_utils import *
from .utils import FkLayout

def dot(a,b):
    if isinstance(a, Twist) and isinstance(b, Wrench):
        return tf.reduce_sum(a.vel*b.force, -1) + tf.reduce_sum(a.rot*b.moment, -1)
    elif isinstance(a, Wrench) and isinstance(b, Twist):
        return tf.reduce_sum(b.vel*a.force, -1) + tf.reduce_sum(b.rot*a.moment, -1)

class Rotation(object):
    def __init__(self, R):
        self.R = R
        self._R_inv = None
        self.is_batch = R.shape.ndims==3

    @property
    def R_inv(self):
        if self._R_inv is None:
            if self.is_batch:
                self._R_inv = tf.transpose(self.R, perm=(0,2,1))
            else:
                self._R_inv = tf.transpose(self.R)
        return self._R_inv

    def inv(self, other=None):
        if other is None:
            return Rotation(R=self.R_inv)

        elif isinstance(other, Twist):
            rot = matvecmul(self.R_inv, other.rot)
            vel = matvecmul(self.R_inv, other.vel)
            return Twist(tf.concat([vel, rot], -1))

        elif isinstance(other, Wrench):
            force = matvecmul(self.R_inv, other.force)
            moment = matvecmul(self.R_inv, other.moment)
            return Wrench(f=force,m=moment)
        elif isinstance(other, tf.Tensor):
            return matvecmul(self.R_inv, other)
        else:
            raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, Twist):
            rot = matvecmul(self.R, other.rot)
            vel = matvecmul(self.R, other.vel)
            return Twist(tf.concat([vel, rot], -1))

        elif isinstance(other, Wrench):
            force = matvecmul(self.R, other.force)
            moment = matvecmul(self.R, other.moment)
            return Wrench(f=force,m=moment)
        elif isinstance(other, tf.Tensor):
            return matvecmul(self.R, other)
        elif isinstance(other, Rotation):
            return matmatmul(self.R, other.R)
        else:
            raise NotImplemented

class Wrench(object):
    def __init__(self, f=None, m=None, w=None):
        """

        :param f: Forces
        :param m: Moments
        :param w: Wrench
        """
        if w is None:
            self.force = f
            self.moment = m
            self.w = tf.concat([f, m], -1)
            self.is_batch = f.shape.ndims == 2
        else:
            self.is_batch = w.shape.ndims == 2
            self.w = w
            self.force = w[...,:3] if self.is_batch else w[:3]
            self.moment = w[..., 3:] if self.is_batch else w[3:]

    def make_batch(self, batch_size=1):
        return Wrench(w=tf.tile(self.w[None], (batch_size, 1)))

    def __add__(self, other):
        if isinstance(other, Wrench):
            return Wrench(f=other.force+self.force, m=other.moment+self.moment)

    def __sub__(self, other):
        if isinstance(other, Wrench):
            return Wrench(f=-other.force+self.force, m=-other.moment+self.moment)

    def __mul__(self, other):
        if isinstance(other, tf.Tensor) or isinstance(other, float):
            return Wrench(w=other*self.w)
        else:
            raise NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Twist):
            f = tf.linalg.cross(other.rot, self.force)
            m = tf.linalg.cross(other.rot, self.moment) + tf.linalg.cross(other.vel, self.force)
            return Wrench(f=f, m=m)
        else:
            raise NotImplemented

class Twist(object):
    def __init__(self, dx=tf.zeros(6)):
        self.dx = dx

    @property
    def is_batch(self):
        return self.dx.shape.ndims == 2

    def make_batch(self, batch_size=1):
        return Twist(tf.tile(self.dx[None], (batch_size,1)))

    def dx_mat(self, m, layout=FkLayout.xm):
        """
        https://en.wikipedia.org/wiki/Angular_velocity
        :param m:
        :return:
        """
        w = angular_vel_tensor(self.rot)
        dm_dphi = matmatmul(w, m)

        if layout is FkLayout.xmv:
            if self.is_batch:
                return tf.concat(
                    [self.vel, tf.reshape(tf.transpose(dm_dphi, perm=(0, 2, 1)), (-1, 9))], axis=1
                )
            else:
                return tf.concat(
                    [self.vel, tf.reshape(tf.transpose(dm_dphi, perm=(1, 0)), (9, 	))], axis=0
                )

        else:
            if self.is_batch:
                return tf.concat(
                    [self.vel, tf.reshape(dm_dphi, (-1, 9))], axis=1
                )
            else:
                return tf.concat(
                    [self.vel, tf.reshape(dm_dphi, (9, 	))], axis=0
                )

    @property
    def vel(self):
        if self.is_batch:
            return self.dx[:, :3]
        else:
            return self.dx[:3]

    @property
    def rot(self):
        if self.is_batch:
            return self.dx[:, 3:]
        else:
            return self.dx[3:]

    def ref_point(self, v):
        if v.shape.ndims > self.rot.shape.ndims:
            rot = self.rot[None] * (tf.zeros_like(v) + 1.)
            vel = self.vel + tf.linalg.cross(rot, v)
        elif v.shape.ndims < self.rot.shape.ndims:
            n = self.rot.shape[0].value
            vel = self.vel + tf.linalg.cross(self.rot, v[None] * tf.ones((n, 1)))
            rot = self.rot
        else:
            vel = self.vel + tf.linalg.cross(self.rot, v)
            rot = self.rot

        if self.is_batch or v.shape.ndims == 2:
            return Twist(tf.concat([vel, rot], 1))
        else:
            return Twist(tf.concat([vel, rot], 0))

    def __add__(self, other):
        if isinstance(other, Twist):
            return Twist(dx=self.dx+other.dx)

    def __sub__(self, other):
        if isinstance(other, Twist):
            return Twist(dx=self.dx-other.dx)


    def __mul__(self, other):
        if isinstance(other, Frame):
            raise NotImplementedError

        elif isinstance(other, tf.Tensor):

            rot = matvecmul(other, self.rot)
            vel = matvecmul(other, self.vel)
            if self.is_batch or rot.shape.ndims == 2:
                return Twist(tf.concat([vel, rot], 1))
            else:
                return Twist(tf.concat([vel, rot], 0))
        elif isinstance(other, Twist):
            return Twist(tf.concat([tf.linalg.cross(self.rot, other.vel) + tf.linalg.cross(self.vel, other.rot),
                             tf.linalg.cross(self.rot, other.rot)], -1))
        else:
            raise NotImplemented



class Frame(object):
    def	__init__(self, p=None, m=None, batch_shape=None):
        """
        :param p:
            Translation vector
        :param m:
            Rotation matrix
        """

        if batch_shape is None:
            p = tf.zeros(3) if p is None else p
            m = tf.eye(3) if m is None else m
        else:
            p = tf.zeros((batch_shape, 3)) if p is None else p
            m = tf.eye(3, batch_shape=(batch_shape, )) if m is None else m

        if isinstance(m, tf.Variable): _m = tf.identity(m)
        else: _m = m

        self.p = p
        if isinstance(m, Rotation):
            self.m = m.R
            self.M = m
        elif isinstance(m, tf.Tensor):
            self.m = m
            self.M = Rotation(m)
        self._M_inv = None
        self._m_inv = None

    @property
    def is_batch(self):
        return self.m.shape.ndims == 3

    def make_batch(self):
        return Frame(m = self.m[None], p = self.p[None] )

    @property
    def m_inv(self):
        if self._m_inv is None:
            self._m_inv = self.M_inv.R
        return self._m_inv

    @property
    def M_inv(self):
        if self._M_inv is None:
            self._M_inv = self.M.inv()
        return self._M_inv

    @property
    def xm(self):
        """
        Position and vectorized rotation matrix
        (order : 'C' - last index changing the first)
        :return:
        """
        if self.is_batch:
            return tf.concat([self.p,  tf.reshape(self.m, [-1, 9])], axis=1)
        else:
            return tf.concat([self.p,  tf.reshape(self.m, [9])], axis=0)

    @property
    def xmv(self):
        """
        Position and vectorized rotation matrix
        (order : 'C' - last index changing the first)
        :return:
        """
        if self.is_batch:
            return tf.concat([self.p,  tf.reshape(tf.transpose(self.m, perm=(0, 2, 1)), [-1, 9])], axis=1)
        else:
            return tf.concat([self.p,  tf.reshape(tf.transpose(self.m, perm=(1, 0)), [9])], axis=0)

    @property
    def xq(self):
        """
        Position and Quaternion
        :return:
        """
        raise NotImplementedError

    def inv(self, other=None):
        if other is None:
            if self.is_batch:
                return Frame(p=-(self.M_inv * self.p),
                             m=self.M_inv)
            else:
                p = tf.expand_dims(self.p, 1)
                return Frame(p=-tf.matmul(self.m_inv, p)[:,0],
                         m=self.M_inv)

        elif isinstance(other, tf.Tensor): # if vector
            return self.M_inv * (other-self.p)

        elif isinstance(other, Wrench):
            force = self.M_inv * other.force
            moment = self.M_inv * (other.moment - tf.linalg.cross(self.p, other.force))
            return Wrench(f=force, m=moment)

        elif isinstance(other, Twist):
            rot = self.M_inv * other.rot
            vel = self.M_inv * (other.vel - tf.linalg.cross(self.p, other.rot))
            return Twist(tf.concat([vel, rot], -1))
        else:
            raise NotImplemented


    def __mul__(self, other):

        if isinstance(other, tf.Tensor) or isinstance(other, tf.Variable):
            if other.shape[-1]== 3: # only position
                if self.is_batch:
                    return tf.einsum('aij,bj->abi', self.m, other) + self.p[:, None]
                else:
                    return tf.einsum('ij,bj->bi', self.m, other) + self.p[None]
            else:
                raise NotImplementedError('Only position supported yet')

        elif isinstance(other, Twist):
            rot = self.M * other.rot
            vel = self.M * other.vel + tf.linalg.cross(self.p, rot)
            return Twist(tf.concat([vel, rot], -1))

        elif isinstance(other, Wrench):
            force = self.M * other.force
            moment = self.M * other.moment + tf.linalg.cross(self.p, force)
            return Wrench(f=force,m=moment)

        elif isinstance(other, Frame):
            m = self.M * other.M
            p = self.M * other.p + self.p
            return Frame(m=m, p=p)

        else:
            return NotImplemented





