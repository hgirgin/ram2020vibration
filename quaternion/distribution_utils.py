from .quaternion import *
d = 4
const = (2 * pi) ** (-d / 2)

def q_mvn_pdf(x, mu, cov):
    mu_ = Quaternion(mu)
    det = tf.math.exp(tf.linalg.slogdet(cov)[1])
    const1 = const * (det ** 0.5)
    diff = x - mu_
    quad = -0.5 * (tf.reduce_sum(diff * tf.linalg.einsum('ij,kj->ki', cov, diff), -1))
    exp_part = tf.math.exp(quad)
    return const1 * exp_part

const_log = (-d/2)*tf.math.log(2*pi)

def q_mvn_logpdf(x, mu, cov):
    mu_ = Quaternion(mu)
    log_det = tf.linalg.slogdet(cov)[1]
    const2 = -0.5*log_det
    diff = x - mu_
    quad = -0.5*(tf.reduce_sum(diff*tf.linalg.einsum('ij,kj->ki',cov,diff), -1))
    return const_log + const2 + quad

def q_sample(mu, cov, sample_size=1):
    # _s = tf.random.uniform(maxval=1., shape=(sample_size,4))
    _s = tf.random.normal( shape=(sample_size,4))
    if mu.ndim == 1:
        _mu = tf.tile(mu[None], (sample_size, 1))
    elif mu.ndim == 2 and mu.shape[0] > 1:
        _mu = tf.tile(mu, (sample_size, 1))
    else:
        _mu = mu
    eig_vals, eig_vecs = tf.linalg.eigh(cov)
    cov_sqrt = eig_vecs@tf.linalg.diag(tf.math.sqrt(eig_vals))

    s = Quaternion(_mu).exp_map(tf.einsum('ij, kj->ki', cov_sqrt, _s))
    return s
