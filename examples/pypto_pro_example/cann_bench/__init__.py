from cann_bench.rms_norm import rms_norm as _rms_norm_impl

def rms_norm(x, gamma, epsilon=1e-6):
    return _rms_norm_impl(x, gamma, epsilon=epsilon)
