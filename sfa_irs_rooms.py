from iear import spherical_image_sources, schroeder_frequency, add_wgn, \
    mixing_time, horizontal_image_sources
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.signal import lfilter, firwin

import sofa
import sys
import time

# NOTE:
# we use azimuth/colatitude convention, although some varibales declare elevation

# https://stackoverflow.com/questions/1465146/how-do-you-determine-a-processing-time-in-python/14739514
def tic():
    global _start_time
    _start_time = time.time()


def tac():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


_start_time = time.time()
print(sys.executable)

flag_parameter_test_set = True
flag_print_ism_table = True
flag_calc_ir = True
flag_plot_ir = True
flag_firlp = True

###############################################################################
# TBD:
# wie speichern (IRs, Plots und Ground Truth Reflections)
# wie Parameter Set aufstellen, so dass Hamilton über Nacht sinnvoll rechnen kann
# TBD: mic array unzulänglichkeiten phi_ideal +- angle  offset


###############################################################################
# general stuff
fs = 32000  # sampling frequency in Hz
Nt = fs // 4  # number of time-domain samples, must span at least +-r/c and initial time delay
f = np.fft.rfftfreq(Nt, 1 / fs)  # Hz
f[0] = f[1]  # DC treatment
w = 2 * np.pi * f

print('set roughly the order Osf=', (2 * np.pi * fs / 2) / 343 * 0.1,
      'for 343 m/s and 0.1m radius')
print('for aliasing free IRs')

O_sf = 30  # modal order of incident sound field, ensure resolution up to fs/2
max_order_ism = 0  # image source model, found per room

Nmic = 360 // 180
thetamic = np.zeros(Nmic) + np.pi / 2
phimic = np.arange(Nmic) * 2 * np.pi / Nmic
weightmic = np.zeros(Nmic) + 2 * np.pi / Nmic
print('theta_mic:', thetamic * 180 / np.pi, 'deg')
print('phi_mic:', phimic * 180 / np.pi, 'deg')
print('weight_mic:', weightmic, 'rad')

###############################################################################
# lowpass FIR
# for odd numtaps -> linphase FIR type 1
# for even numtaps -> linphase FIR type 2
# -60 dB highest stopband ripple
firlp = firwin(numtaps=512, cutoff=fs * 0.325, fs=fs,
               window=('kaiser', 0.1102 * (60 - 8.7)))

###############################################################################
# parameter variation
room_x = np.arange(3, 10, 1)  # in m,  Room Length
room_y = np.arange(3, 10, 1)  # in m, Room Width
reflection_coeff = np.arange(-3,
                             4) / 20 + 3 / 4  # Room Reflection Coefficients for 4 Walls
speed_of_sound = 331 + 0.6 * np.arange(10, 30, 2)  # in m/s, Speed of Sound
ls_x = np.arange(1) / 10 + 3  # in m, Source Position x
ls_y = np.arange(1) / 10 + 3  # in m, Source Position y
mic_r = np.arange(4, 7) / 50  # in m, Mic Array Radius
mic_x = np.arange(5) / 10 + 3  # in m, Mic Array Position x
mic_y = np.arange(5) / 10 + 3  # in m, Mic Array Position y
snr = np.arange(0, 40, 10)  # in dB, Peak to Noise Ratio

if flag_parameter_test_set:
    # single debug test set
    room_x = [11]
    room_y = [13]
    reflection_coeff = [1]
    speed_of_sound = [333.3333]
    ls_x = [7]  # right
    ls_y = [6]
    mic_r = [0.1]
    mic_x = [3 + 1.2345]  # left
    mic_y = [np.exp(1)]
    snr = [200]

print('min/max parameters:')
print('room_x', np.min(room_x), np.max(room_x))
print('room_y', np.min(room_y), np.max(room_x))
print('reflection_coeff', np.min(reflection_coeff), np.max(reflection_coeff))
print('speed_of_sound', np.min(speed_of_sound), np.max(speed_of_sound))
print('ls_x', np.min(ls_x), np.max(ls_x))
print('ls_y', np.min(ls_y), np.max(ls_y))
print('mic_r', np.min(mic_r), np.max(mic_r))
print('mic_x', np.min(mic_x), np.max(mic_x))
print('mic_y', np.min(mic_y), np.max(mic_y))
print('snr', np.min(snr), np.max(snr))
print()

gen_room_counter = 0
for rx, ry, rc, c, lx, ly, mr, mx, my, sn in [
    (rx, ry, rc, c, lx, ly, mr, mx, my, sn)
    for rx in room_x
    for ry in room_y
    for rc in reflection_coeff
    for c in speed_of_sound
    for lx in ls_x
    for ly in ls_y
    for mr in mic_r
    for mx in mic_x
    for my in mic_y
    for sn in snr]:
    tic()
    gen_room_counter += 1  #
    print('##########\nroom %d\n##########' % gen_room_counter)

    ###########################################################################
    k = w / c  # rad/m

    ###########################################################################
    # mic array characteristics
    # mr = mr  # radius of array
    tarray = 2 * mr / c  # in s
    array_type = 'rigid'  # configuration of array
    mic_array_pos = [mx, my, np.sqrt(rx * ry) / 2]  # mic array origin position
    SNR = sn  # peak-to-noise level ratio of microphone impulse responses

    rmic = np.zeros(Nmic) + mr
    xmic = np.array([phimic, thetamic, rmic])
    # print('2D Mic Array Horizontal Equi-Angular Circle, dim xmic: (3, Nmic)=',
    #      xmic.shape, ', Nmic:', Nmic)
    # print('r_mic:', rmic, 'm')

    ###########################################################################
    # room characteristics
    # rectangular room, e.g. length x, width y, height z = sqrt(x*y) !!!
    # this is important to get the mixing time (which does not consider absorption)
    L = [rx, ry, np.sqrt(rx * ry)]  # sqrt(x*y) is chosen to have a closed room
    if L[2] < 2.4:  # if unrealistic low ceiling
        L[2] = 2.4
    # reflection coefficient per wall, no floor/no ceiling (i.e. fully absorbing):
    coeffs = [rc, rc, rc, rc, 0.0, 0.0]
    print('Schroeder frequency of room: %5.1f Hz' %
          schroeder_frequency(L, coeffs))

    ###########################################################################
    # 0...50% or 1...95% mixing time according to Lindau/Kosanke, JAES
    tmix = mixing_time(L)[1]  # in ms

    ###########################################################################
    # set up point source characteristics
    source_pos = [lx, ly, np.sqrt(
        rx * ry) / 2]  # make sure same height! as mic_array_pos
    print('rec2src dist =  %3.2f m' % norm(
        np.array(mic_array_pos) - np.array(source_pos)))

    max_order_ism,\
    ref_azi_is,\
    ref_elev_is,\
    ref_r_is,\
    ref_t_is = horizontal_image_sources(source_pos=source_pos,
                                        mic_array_pos=mic_array_pos,
                                        L=L,
                                        Rmic=mr,
                                        c=c,
                                        fs=fs,
                                        flag_print_ism_table=flag_print_ism_table)

    ###########################################################################
    # get spherical_image_sources onto rigid sphere scatterer
    # where we only use the equator mics
    # Furthermore please note:
    # !!! HACK !!! in order to get NO image sources from ceiling/floor
    # we set height L[2] very, very large
    # thus we assume the room definition and mixing time from above
    # but in the IRs we only consider the horizontal stuff
    # not nice and also completely out of reality
    # but as a workaround for proof of concept this is considered to be feasible
    if flag_calc_ir:
        L[2] = 1e6
        source_pos[2] = 1e6 // 2
        mic_array_pos[2] = 1e6 // 2
        Y = spherical_image_sources(O_sf, k, phimic, thetamic, mr,
                                    source_pos, mic_array_pos, L,
                                    max_order_ism, coeffs, array_type)
        if SNR is not None:  # add uncorrelated noise in freq domain
            Y, _ = add_wgn(Y, SNR)  # checked: ok
        y = np.transpose(np.fft.irfft(Y, axis=0))
        if flag_firlp:
            y = lfilter(firlp, 1, y, axis=-1)
        t = np.arange(0, Nt) / fs
        print('dim Y: (frequencies, mics):', Y.shape)
        print('dim y: (mics, samples)=', y.shape)
        # dim Y: (frequencies, mics): (4001, 2) 4001 frequency bins, 2 mics
        # dim y: (mics, samples) = (2, 8000) 2 mics, 8000 samples
    if flag_plot_ir:
        for n in range(Nmic):
            ir_db = 20 * np.log10(np.abs(y[n, :]))
            ir_db = ir_db - np.max(ir_db)
            plt.plot(t * 1000, ir_db)
            plt.xlabel('t / ms')
            plt.ylabel('level of yIR / dB')
            plt.xlim(0, tmix)
            plt.ylim((-50, 10))
            plt.xticks(np.arange(0, tmix, 6))
            plt.grid(True)
        plt.show()
    tac()

print('##########################')
print('number of generated rooms:', gen_room_counter)
print('##########################')
