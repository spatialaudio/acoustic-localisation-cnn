import sofa as _sofa
import numpy as _np
import scipy.signal as _sig
import soundfile as _sf
import netCDF4 as _ncdf


def spatialise_signal(ir_file: str,
                      output_file: str,
                      signal: str = None,
                      noise_length: int = None
                      ):

    # Open IR File
    ir = _sofa.Database.open(ir_file)
    fs_ir = ir.Data.SamplingRate.get_values()  # sampling frequency in Hz

    # Some dimensions
    M = ir.Dimensions.M
    E = ir.Dimensions.E
    Nh = ir.Dimensions.N

    # handling of signals
    if (noise_length is None) and (signal is None):
        raise ValueError(
            'if no signal is provided, signal_length has to be defined'
        )
    else:
        Nin = noise_length

    if signal is None:
        if (noise_length is None):
            raise ValueError(
                'if no signal is provided, signal_length has to be defined'
            )
        else:
            Nin = noise_length
    else:
        sig_in, fs_sig = _sf.read(signal)
        if not (fs_ir == fs_sig):
            raise RuntimeWarning(
                'Sampling Frequency of IR ({0:%.2f}) and Signal ({1:%.2f}) do not match!'.format(
                    fs_ir, fs_sig)
            )
        Nin = sig_in.shape[0]

    Nout = Nin + Nh - 1
    if not signal is None:
        spec_in = _np.fft.rfft(sig_in[:, _np.newaxis], axis=0, n=Nout)

    # used low-level netcdf4 API to create new copy of SOFA file
    with _ncdf.Dataset(ir_file) as src, _ncdf.Dataset(output_file, "w") as dst:
        # copy global attributes all at once via dictionary
        dst.setncatts(src.__dict__)
        # copy dimensions
        for name, dimension in src.dimensions.items():
            dim_len = len(dimension) if not name == "N" else Nin
            dst.createDimension(
                name, (dim_len if not dimension.isunlimited() else None)
            )
        # copy all variables
        for name, variable in src.variables.items():
            dst.createVariable(name, variable.datatype, variable.dimensions)
            if not "N" in variable.dimensions:
                dst[name][:] = src[name][:]
            # copy variable attributes all at once via dictionary
            dst[name].setncatts(src[name].__dict__)

    # convolution
    out = _sofa.Database.open(output_file, 'r+')
    for kk in range(E):
        for ii in range(M):
            if signal is None:
                sig_in = _np.random.normal(0, 1, (1, Nin))  # dry source signal
                spec_in = _np.fft.rfft(sig_in, axis=1, n=Nout)

            sig_h = ir.Data.IR.get_values(
                indices={"M": ii, "E": kk},
                dim_order=("R", "N")
            )

            spec_h = _np.fft.rfft(sig_h, axis=1, n=Nout)
            sig_out = _np.fft.irfft(spec_in*spec_h, axis=1, n=Nout)
            sig_out = sig_out[:,:Nin]

            # for channel-joint Gaussian normalisation
            rms = _np.sqrt(_np.mean(sig_out**2))

            # write data
            out.Data.IR.set_values(
                sig_out / rms,
                indices={"M": ii, "E": kk},
                dim_order=("R", "N")
            )

    out.close()
    ir.close()


def diffuse_noise(ir_file: str,
                  output_file: str,
                  noise_length: int = 1024,
                  ):

    # Open IR File
    ir = _sofa.Database.open(ir_file)

    # Some dimensions
    Nout = noise_length + ir.Dimensions.N - 1

    # convolution
    sig_out = _np.zeros((ir.Dimensions.R, Nout))
    for kk in range(ir.Dimensions.E):
        for ii in range(ir.Dimensions.M):            
            sig_noise = _np.random.normal(0, 1, (1, noise_length))
            spec_noise = _np.fft.rfft(sig_noise, axis=1, n=Nout)

            sig_h = ir.Data.IR.get_values(
                indices={"M": ii, "E": kk},
                dim_order=("R", "N")
            )

            spec_h = _np.fft.rfft(sig_h, axis=1, n=Nout)
            sig_out += _np.fft.irfft(spec_noise*spec_h, axis=1, n=Nout)

    # close sofa file
    ir.close()

    # for channel-joint Gaussian normalisation
    rms = _np.sqrt(_np.mean(sig_out**2))

    # save numpy array
    _np.save(output_file, sig_out/rms, allow_pickle=False)


def pressure_plane_wave_open_sphere(src_sig, R, Nmic, Nphi, fs, c):
    Nsig = src_sig.size

    # plane wave directions
    phi_pw = _np.linspace(0, 2 * _np.pi, Nphi, endpoint=False)

    # microphone angles
    phi_mic = _np.linspace(0, 2 * _np.pi, Nmic, endpoint=False)

    # microphone signals
    signal = _np.zeros((Nsig, Nmic, Nphi))
    for ii in range(Nphi):
        for jj in range(Nmic):
            delay = R / c * _np.cos(phi_pw[ii] - phi_mic[jj])
            shift = int(_np.round(- delay * fs))

            signal[:, jj, ii] = _np.roll(src_sig, shift)

    pos = _np.array(
        [phi_pw, _np.repeat(_np.pi/2.0, Nphi), _np.repeat(1, Nphi)]
    )

    rec_pos = _np.array(
        [phi_mic, _np.repeat(_np.pi/2.0, Nmic), _np.repeat(R, Nmic)]
    )

    return signal, pos, rec_pos


def hn2_poly(n):
    """Bessel polynomial of n-th order.
    Polynomial that characterizes the spherical Hankel functions.
    The coefficients are computed by using the recurrence relation.
    The returned array has a length of n+1. The first coefficient is always 1.

    Parameters
    ----------

    n : int
        Bessel polynomial order.

    """
    beta = _np.zeros(n + 1)
    beta[n] = 1
    for k in range(n-1, -1, -1):
        beta[k] = beta[k+1] * (2*n-k) * (k+1) / (n-k) / 2
    return beta


def decrease_hn2_poly_order_by_one(beta):
    """Bessel polynomial of order decreased by 1.
    """
    n = len(beta)-1
    alpha = _np.zeros(n)
    for k in range(n-1):
        alpha[k] = beta[k+1] * (k+1) / (2*n-k-1)
    alpha[-1] = 1
    return alpha


def derivative_hn2_poly(n):
    """Polynomial characterizing the derivative of the spherical Hankel func.
    """
    gamma = hn2_poly(n+1)
    gamma[:-1] -= n * decrease_hn2_poly_order_by_one(gamma)
    return gamma

# from
# https://github.com/spatialaudio/time-domain-modal-processing/tree/master/python


def matchedz_zpk(s_zeros, s_poles, s_gain, fs):
    """Matched-z transform of poles and zeros.
    """
    z_zeros = _np.exp(s_zeros / fs)
    z_poles = _np.exp(s_poles / fs)
    z_gain = s_gain
    return z_zeros, z_poles, z_gain


# from
# https://github.com/spatialaudio/time-domain-modal-processing/tree/master/python

def s2z_zpk(s_zeros, s_poles, s_gain, s2z, fs, f0):
    z_zeros, z_poles, z_gain = s2z(s_zeros, s_poles, s_gain, fs=fs)
    z_gain *= _np.abs(_sig.freqs_zpk(s_zeros, s_poles, s_gain, worN=[2*_np.pi*f0])[1])\
        / _np.abs(_sig.freqz_zpk(z_zeros, z_poles, z_gain, worN=[f0], fs=fs)[1])
    return z_zeros, z_poles, z_gain
