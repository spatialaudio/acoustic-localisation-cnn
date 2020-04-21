# %%

import tools.audio
import numpy as np
import scipy.signal as sig
import scipy.special as spec
import matplotlib.pyplot as plt

# %%
# parameters
c = 343  # speed of sound in m/s
fs = 44100  # sampling frequency in Hz
M = 10  # modal order
N = 512  # ir length
R = 0.1
r = 1.0
tauR = R/c
taur = r/c

# %%
# radial filters prototypes

hn_p = [tools.audio.hn2_poly(i) for i in range(M+1)]
hnprime_p = [tools.audio.derivative_hn2_poly(i) for i in range(M+1)]

hn_zeros = [np.roots(hn_p[i][::-1]) for i in range(M+1)]
hnprime_zeros = [np.roots(hnprime_p[i][::-1]) for i in range(M+1)]

# %%

fig, axs = plt.subplots(ncols=2, nrows=2)

for m in range(M+1):

    # Ground Truth Transfer Function
    f, _ = sig.freqz(1.0, worN=1024, fs=fs)  # just to get the f-axis
    omega = 2*np.pi*f

    hn = spec.spherical_jn(m, omega*taur)\
        - 1j*spec.spherical_yn(m, omega*taur)

    hnprime = spec.spherical_jn(m, omega*tauR, derivative=True)\
        - 1j*spec.spherical_yn(m, omega*tauR, derivative=True)

    Hgt = - c / (omega*R**2) * hn / hnprime
    Hgt *= np.exp(1j*omega*(taur - tauR))  # compensate delay
    Hgt[0] = Hgt[1]  # deal with NaNs

    # Transfer Function of IIR-Implementation
    z_s = hn_zeros[m] / taur
    p_s = hnprime_zeros[m] / tauR

    # iterate over possible method to get from Laplace into z-domain
    for sdx, s2z_method in enumerate([sig.bilinear_zpk, tools.audio.matchedz_zpk]):

        # transform zeros/poles from Laplace into z-domain
        z_z, p_z, k_z = tools.audio.s2z_zpk(
            s_zeros=z_s, s_poles=p_s, s_gain=c/(r*R), s2z=s2z_method, fs=fs, f0=1
        ) 

        sos = sig.zpk2sos(z_z, p_z, k_z)

        [f, Hz_freqz] = sig.sosfreqz(sos, worN=1024, fs=fs)          

        axs[sdx][0].semilogx(omega, 20*np.log10(abs(Hgt)),
                        linewidth=3, color='gray')
        axs[sdx][0].semilogx(omega, 20*np.log10(abs(Hz_freqz)), '--')

        axs[sdx][1].semilogx(omega, np.unwrap(np.angle(Hgt)), linewidth=2, color='gray')
        axs[sdx][1].semilogx(omega, np.unwrap(np.angle(Hz_freqz)), '--')

plt.show()
