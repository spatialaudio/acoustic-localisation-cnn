# %%
from time import strftime
import tools
import numpy as np
import scipy.signal as sig
import scipy.special as spec
import matplotlib.pyplot as plt
import sofa
import os

# %%
# parameters
c = 343  # speed of sound in m/s
fs = 44100  # sampling frequency in Hz
N = 128  # ir length
filename = 'tmp/train/RIGID_SPHERE_anechoic.sofa'
create_sofa = True  # create .sofa of impulse responses

# array parameters
R = 0.085  # radius of rigid sphere mic array in m
Nmic = 2  # number of sensors,
# Nmic = 4 is comparable with Lebedev3D grid Nmic=14 used for ICA Aachen 2019
phimic = np.array([np.pi/2, -np.pi/2])  # left first
thetamic = np.repeat([np.pi/2], Nmic)

M = np.int(np.ceil(2 * np.pi * fs / 2 / c * R))  # modal order
print('modal order %d' % M)

# source parameters
Nr = 1
Ndirs = 360
Nmeasure = Nr * Ndirs
# Rs = np.arange(Nr)/100*(33+1/3) + 2  # 1 m steps, starting from 2 m
Rs = [2.0]
print(Rs)
phis = np.linspace(np.pi, -np.pi, Ndirs, endpoint=False)[::-1]
thetas = np.repeat([np.pi/2], Ndirs)

# %%
# radial filters prototypes

hn_p = [tools.audio.hn2_poly(i) for i in range(M + 1)]
hnprime_p = [tools.audio.derivative_hn2_poly(i) for i in range(M + 1)]

hn_zeros = [np.roots(hn_p[i][::-1]) for i in range(M + 1)]
hnprime_zeros = [np.roots(hnprime_p[i][::-1]) for i in range(M + 1)]

# %%
impulse = np.zeros(N)
impulse[0] = 1
# for even N -> linphase FIR type 2, for odd N -> linphase FIR type 1
# -60 dB highest stopband ripple
if False:
    impulse = sig.firwin(numtaps=N, cutoff=fs * 0.3125, fs=fs,
                         window=('kaiser', 0.1102 * (60 - 8.7)))

irs = np.zeros((N, Nmic, Nmeasure))

# method to compute transform zeros/poles from Laplace into z-domain
# s2z_method = tools.audio.matchedz_zpk
s2z_method = sig.bilinear_zpk

for rdx, r in enumerate(Rs):

    # radial filters for rs
    h_rad = np.zeros((N, M + 1))
    for m in range(M + 1):
        z_s = hn_zeros[m] * c / r
        p_s = hnprime_zeros[m] * c / R

        # transform zeros/poles from Laplace into z-domain
        z_z, p_z, k_z = tools.audio.s2z_zpk(
            s_zeros=z_s,
            s_poles=p_s,
            s_gain=c / (r * R),
            s2z=s2z_method,
            fs=fs, f0=1
        )

        # second-order sections
        sos = sig.zpk2sos(z_z, p_z, k_z)

        h_rad[:, m] = sig.sosfilt(sos, impulse, axis=0)

    for ddx, (theta, phi) in enumerate(zip(thetas, phis)):
        cosTheta = np.sin(theta) * np.sin(thetamic) * np.cos(phi - phimic) \
            + np.cos(theta) * np.cos(thetamic)

        weight_modal = np.vstack(
            [(2 * m + 1) * spec.eval_legendre(m, cosTheta[np.newaxis, :]) for m
             in range(M + 1)]
        )

        irs[:, :, rdx * Ndirs + ddx] = h_rad @ weight_modal

    irs[:, :, rdx * Ndirs:(rdx + 1) * Ndirs] *= 1.0 / (4 * np.pi)

# %%
# python-sofa 0.2.0 required
try:
    os.remove(filename)
except Exception:
    pass

HRIR = sofa.Database.create(
    filename,
    "SimpleFreeFieldHRIR",
    {
        "N": N,
        "M": Nmeasure,
        "R": Nmic,
        "E": 1
    }
)

HRIR.Metadata.set_attribute(
    'AuthorContact', 'Fiete Winter (fiete.winter@uni-rostock.de)')
HRIR.Metadata.set_attribute('Organization', 'INT/IEF/University of Rostock')
HRIR.Metadata.set_attribute(
    'Title', 'HRIR for Sound-rigid spherical Head'.format(R))
HRIR.Metadata.set_attribute('Comment', 'experimental')
HRIR.Metadata.set_attribute('DatabaseName', 'daga2020_doa_iear_dnn')

HRIR.Metadata.set_attribute(
    'ListenerDescription',
    'Sound-rigid spherical Head ({0:2.3f} m radius), coordinates are absolute'.format(R)
)
HRIR.Metadata.set_attribute(
    'ReceiverDescription',
    'Ideal 3D monopole receiver, coordinates are relative'
)
HRIR.Metadata.set_attribute(
    'SourceDescription',
    'Meta object, coordinates are absolute'
)
HRIR.Metadata.set_attribute(
    'EmitterDescription',
    'Ideal monopole point source, coordinates are relative'
)

HRIR.Metadata.set_attribute('RoomType', 'freefield')
HRIR.Metadata.set_attribute('RoomDescription',
                            'raw data for DNN training set')

HRIR.Metadata.set_attribute('License', 'CCBY4.0')
HRIR.Metadata.set_attribute('DateCreated', strftime("%Y-%m-%d %H:%M:%S"))
HRIR.Metadata.set_attribute('DateModified', 'not modified')

HRIR.Listener.initialize(['Position', 'View', 'Up'])

rec_pos = np.vstack(
    [phimic,
     np.pi/2.0 - thetamic,
     R * np.ones_like(phimic)
     ]
)
HRIR.Receiver.initialize(['Position'])
HRIR.Receiver.Position.set_system(
    ctype='spherical', cunits='rad, rad, meter'
)
HRIR.Receiver.Position.set_values(
    rec_pos, dim_order=('C', 'R'), indices={'M': 0}
)

HRIR.Source.initialize(['View', 'Up'], ['Position'])

# we need this data alignment when irs[:, :, rdx * Ndirs + ddx]:
pos = np.array(
    [np.tile(phis, Nr),
     np.tile(np.pi/2.0 - thetas, Nr),
     np.repeat(Rs, Ndirs)
     ]
)

HRIR.Source.Position.set_system(
    ctype='spherical', cunits='rad, rad, meter'
)
HRIR.Source.Position.set_values(
    pos, dim_order=('C', 'M')
)

HRIR.Emitter.initialize(['Position'])

HRIR.Data.initialize(N)
HRIR.Data.SamplingRate.set_values(fs)
HRIR.Data.IR.set_values(irs, dim_order=('N', 'R', 'M'))

HRIR.Metadata.dump()
HRIR.Dimensions.dump()

HRIR.close()
