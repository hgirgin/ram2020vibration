from .frame import Frame, Twist, Wrench, Rotation
from .utils.tf_utils import *



class RotationalInertia(object):
    def __init__(self, Ixx=None, Iyy=None, Izz=None, Ixy=None, Ixz=None, Iyz=None, I=None):
        self.is_batch=False
        if I is None:
            if not isinstance(Ixx, float):
                if Ixx.shape.ndims == 2:
                    self.is_batch=True
                    row1 = tf.concat([Ixx, Ixy, Ixz],-1)
                    row2 = tf.concat([Ixy, Iyy, Iyz], -1)
                    row3 = tf.concat([Ixz, Iyz, Izz], -1)
                    self.I = tf.concat([row1, row2, row3],0)
            else:
                self.I = tf.constant([[Ixx, Ixy, Ixz],[Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        else:
            self.I = I

    def __mul__(self, other):
        if isinstance(other, float):
            return RotationalInertia(I=other * self.I)
        if isinstance(other, tf.Tensor):
            if other.shape.ndims==0:
                return RotationalInertia(I=other*self.I)
            else:
                raise NotImplemented

    def __add__(self, other):
        if isinstance(other, RotationalInertia):
            return RotationalInertia(I=self.I+other.I)

class RigidBodyInertia(object):
    def __init__(self, m=None, h=None,I_r=None,c=None, I_c=None):
        """
        the arguments are the mass, the vector from the reference point to cog and the rotational inertia in the cog.
        :param m_:
        :param h_:
        :type I_r: RotationalInertia
        :param I_r:
        :param c_:
        :type I_c: RotationalInertia
        :param I_c:
        """

        self.m = m
        self.h = m*c if h is None else h

        if I_c is None:
            self.I_r = I_r
        else:
            self.I_r = RotationalInertia(I=I_c.I-m*(tf.einsum('i,j->ij',c,c)-tf.reduce_sum(c**2,-1)*tf.eye(I_c.I.shape[-1])))

        self.is_batch = self.I_r.I.shape.ndims==3

    def make_batch(self, batch_size=1):
        if not self.is_batch:
            m = tf.ones(batch_size)*self.m
            h = tf.tile(self.h[None], (batch_size,1))
            I = tf.tile(self.I_r.I[None], (batch_size,1,1))
            return RigidBodyInertia(m=m, h=h, I_r=RotationalInertia(I=I))

    def __add__(self, other):
        if isinstance(other, RigidBodyInertia):
            return RigidBodyInertia(m=self.m + other.m, h=self.h + other.h, I_r=RotationalInertia(I=self.I_r.I + other.I_r.I))
        else:
            raise NotImplemented

    def __mul__(self, other):
        if isinstance(other, tf.Tensor):
            if other.shape.ndims == 1: # just a constant
                return RigidBodyInertia(other*self.m, other*self.h, RotationalInertia(I=other*self.I_r.I))
        elif isinstance(other, Rotation):
            m_res = self.m
            h_res = matvecmul(other.R, self.h)
            I_res = RotationalInertia(I=other.R_inv @ self.I_r.I @ other.R)
            return RigidBodyInertia(m_res, h_res, I_res)
        elif isinstance(other, Frame):
            X = other.inv()
            h = self.h
            I = self.I_r.I
            R = X.m
            r = X.p
            TM = other.m
            m = self.m
            batch_size=h.shape[0]

            hmr = h-m[:,None]*r

            rcrosshcross = tf.einsum('ai,aj->aij', h, r)
            hmrcrossrcross = tf.einsum('ai,aj->aij', r, hmr)
            rcrosshcross += - tf.reduce_sum(r * h, -1)[:,None,None] * tf.eye(rcrosshcross.shape[-1], batch_shape=[batch_size])
            hmrcrossrcross += - tf.reduce_sum(hmr * r, -1)[:,None,None] * tf.eye(hmrcrossrcross.shape[-1], batch_shape=[batch_size])

            # Ib = R @ (I + rcrosshcross + hmrcrossrcross) @ tf.linalg.inv(R)
            Ib = X.m_inv @ (I + rcrosshcross + hmrcrossrcross) @ R
            return RigidBodyInertia(m=m, h=matvecmul(TM,hmr), I_r=RotationalInertia(I=Ib))

        elif isinstance(other, Twist):
            rot = other.rot
            vel = other.vel
            h = self.h
            I = self.I_r.I

            f = self.m[:,None]*vel-tf.linalg.cross(h, rot)
            m = matvecmul(I, rot)+tf.linalg.cross(h, vel)
            return Wrench(f=f, m=m)
        else:
            raise NotImplemented

    def cog(self):
        return self.h/self.m[:,None]

    def ref_point(self, p):
        raise NotImplemented


class ArticulatedBodyInertia(object):
    """
    6D Inertia of a articulated body.
    The inertia is defined in a certain reference point and a certain reference base.
    The reference point does not have to coincide with the origin of the reference frame.
    """
    def __init__(self, rbi=None, M=None, I=None, H=None):
        """
        :type rbi: tk.RigidBodyInertia
        :param rbi:
        """
        if rbi is not None:
            self._M = tf.eye(3) * rbi.m
            self._I = rbi.I_r.I
            self._H = skew(rbi.h)
        else:
            assert M is not None
            assert I is not None
            assert H is not None
            self._M = M
            self._I = I
            self._H = H
        self._is_batch = self._M.shape.ndims == 3
        if self._is_batch:
            self._H_T = tf.transpose(self._H, (0,2,1))
        else:
            self._H_T = tf.transpose(self._H)

    @property
    def is_batch(self):
        return self._is_batch

    def __add__(self, other):
        if isinstance(other, ArticulatedBodyInertia):
            return ArticulatedBodyInertia(M=other._M + self._M,
                                          I=other._I * self._I,
                                          H=other._H * self._H)
        elif isinstance(other, RigidBodyInertia):
            return ArticulatedBodyInertia(other) + self
        else:
            raise NotImplemented

    def __sub__(self, other):
        if isinstance(other, ArticulatedBodyInertia):
            return ArticulatedBodyInertia(M=-other._M + self._M,
                                          I=-other._I * self._I,
                                          H=-other._H * self._H)
        elif isinstance(other, RigidBodyInertia):
            return ArticulatedBodyInertia(other) - self

    def __mul__(self, other):
        if isinstance(other, tf.Tensor):
            if other.shape.ndims == 2:
                return ArticulatedBodyInertia(M=other*self._M,
                                              I=other*self._I,
                                              H=other*self._H)
            else: # if rotation class todo
                E = other
                E_T = tf.transpose(E, axis=(0, 2, 1))
                return ArticulatedBodyInertia(M=E_T @ self._M @ E,
                                              H=E_T @ self._H @ E,
                                              I=E_T @ self._I @ E)

        elif isinstance(other, Twist):
            force = matvecmul(self._M, other.vel) + matvecmul(self._H_T, other.rot)
            torque = matvecmul(self._I, other.rot) + matvecmul(self._H, other.vel)
            return Wrench(f=force, m=torque)
        else:
            raise NotImplemented


    def __rmul__(self, other):
        if isinstance(other, Frame):
            inv_frame = other.inv()
            E = inv_frame.m
            E_T = tf.transpose(E, (0, 2, 1))
            rcross = skew_batch(inv_frame.p) if inv_frame.is_batch else skew(inv_frame.p)
            HrM = self._H - rcross @ self._M
            return ArticulatedBodyInertia(M=E @ self._M @ E_T,
                                          H=E @ HrM @ E_T,
                                          I=E @ (self._I - rcross @ self._H_T + HrM @ rcross) @ E_T)
        elif isinstance(other, Rotation):
            E = other.R
            E_T = tf.transpose(E, (0, 2, 1)) if other.is_batch else tf.transpose(E)
            return ArticulatedBodyInertia(M=E_T @ self._M @ E,
                                          H=E_T @ self._H @ E,
                                          I=E_T @ self._I @ E)

        else:
            raise NotImplemented

    def ref_point(self, p):
        rcross = skew(p)
        HrM = self._H - rcross @ self._M
        return ArticulatedBodyInertia(M=self._M,
                                      H=HrM,
                                      I=self._I-rcross@self._H_T+HrM@rcross)