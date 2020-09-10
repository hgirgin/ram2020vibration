import tensorflow as tf
import numpy as np
import scipy as sc

def rot2axis_angle(R):
    value = 0.5*(tf.linalg.trace(R)-1)
    value = tf.clip_by_value(value, -1+1e-6, 1-1e-6) # prevent arccos grad nan
    theta = tf.math.acos(value)[:, None]
    u = unskew(R- tf.transpose(R, (0,2,1)))/(2*tf.math.sin(theta))
    return u, theta

def rot2quat(R):
    u, theta = rot2axis_angle(R)
    return axis_angle2quats(u, theta)


def unskew(R):
    w2 = -R[:, 0, 1]
    w1 = R[:, 0, 2]
    w0 = -R[:, 1, 2]
    return tf.transpose(tf.stack([w0, w1, w2]))

def skew(w):
    n = w.shape[0]
    zeros = tf.zeros((n,))
    return tf.transpose(tf.stack([[zeros, -w[:, 2], w[:, 1]],
                                  [w[:, 2], zeros, -w[:, 0]],
                                  [-w[:, 1], w[:, 0], zeros]] ),(2,0,1))
def omega_quat_left(x):
    n = x.shape[0]
    w = x[:, 0][:, None]
    v = x[:, 1:]

    omega00 = w
    omega_0 = v
    omega0_ = -v
    omega__ = tf.eye(3, batch_shape=(n,))*w[:,None] + skew(v)
    return tf.concat([tf.concat([omega00, omega0_],-1)[:,None], tf.concat([omega_0[..., None], omega__],-1)],1)

def omega_quat_right(x):
    n = x.shape[0]
    w = x[:, 0][:, None]
    v = x[:, 1:]

    omega00 = w # [n, 1]
    omega_0 = v # [n, 3]
    omega0_ = -v # [n, 3]
    omega__ = tf.eye(3, batch_shape=(n,))*w[:,None]- skew(v) # [n, 3, 3]
    return tf.concat([tf.concat([omega00, omega0_],-1)[:,None], tf.concat([omega_0[..., None], omega__],-1)],1)


def rotation_matrix_from_axis_angle(ax, angle):
    """
    Gets rotation matrix from axis angle representation using Rodriguez formula.
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param ax: unit axis defining the axis of rotation
    :param angle: angle of rotation

    Returns
    -------
    :return: R(ax, angle) = I + sin(angle) x ax + (1 - cos(angle) ) x ax^2 with x the cross product.
    """
    utilde = vector_to_skew_matrix(ax)
    return np.eye(3) + np.sin(angle)*utilde + (1 - np.cos(angle))*utilde.dot(utilde)


def vector_to_skew_matrix(q):
    """
    Transform a vector into a skew-symmetric matrix

    Parameters
    ----------
    :param q: vector

    Returns
    -------
    :return: corresponding skew-symmetric matrix
    """
    return np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])

def get_axisangle(d):
    """
    Gets axis-angle representation of a point lying on a unit sphere
    Based on the function of riepybdlib (https://gitlab.martijnzeestraten.nl/martijn/riepybdlib)

    Parameters
    ----------
    :param d: point on the sphere

    Returns
    -------
    :return: axis, angle: corresponding axis and angle representation
    """
    norm = np.sqrt(d[0]**2 + d[1]**2)
    if norm < 1e-6:
        return np.array([0, 0, 1]), 0
    else:
        vec = np.array([-d[1], d[0], 0])
        return vec/norm, np.arccos(d[2])

def axis_angle2quats(u, theta):
    angle = theta/2
    qw = tf.math.cos(angle)
    qv = u*tf.math.sin(angle)
    if isinstance(qw, float):
        return tf.concat([qw[None], qv],-1)
    elif isinstance(qw, tf.Tensor):
        if qw.shape.ndims == 0:
            return tf.concat([qw[None], qv],-1)
        else:
            return tf.concat([qw, qv], -1)


def create_unit_rotations(angle):
    axes = tf.eye(3)
    if isinstance(angle,float):
        angle = tf.tile(tf.constant([angle])[None], (3,1))
    elif angle.ndim == 1: #
        angle = angle[:,None]

    return axis_angle2quats(axes, angle)

def expmap(u, x0):
    """
    This function maps a vector u lying on the tangent space of x0 into the manifold.

    Parameters
    ----------
    :param u: vector in the tangent space
    :param x0: basis point of the tangent space

    Returns
    -------
    :return: x: point on the manifold
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(u) < 2:
        u = u[:, None]

    norm_u = np.sqrt(np.sum(u*u, axis=0))
    x = x0 * np.cos(norm_u) + u * np.sin(norm_u)/norm_u

    x[:, norm_u < 1e-16] = x0

    return x


def logmap(x, x0):
    """
    This functions maps a point lying on the manifold into the tangent space of a second point of the manifold.

    Parameters
    ----------
    :param x: point on the manifold
    :param x0: basis point of the tangent space where x will be mapped

    Returns
    -------
    :return: u: vector in the tangent space of x0
    """
    if np.ndim(x0) < 2:
        x0 = x0[:, None]

    if np.ndim(x) < 2:
        x = x[:, None]

    theta = np.arccos(np.maximum(np.minimum(np.dot(x0.T, x), 1.), -1.))
    u = (x - x0 * np.cos(theta)) * theta/np.sin(theta)

    u[:, theta[0] < 1e-16] = np.zeros((u.shape[0], 1))

    return u