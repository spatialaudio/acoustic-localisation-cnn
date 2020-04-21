import numpy as np

c = 343  # m/s
fs = 48000  # Hz
Rmic = 0.1  # m
t = Rmic/c  # s

# Jitter
block_jitter = 2  # samples

Rmax = ((t + block_jitter/fs) * c)
Rmin = ((t - block_jitter/fs) * c)
print('meaningful radius changes:')
print('Mic Array Radius Variation Due to Jitter, Rmax = %5.4f m' % Rmax)
print('Mic Array Radius Variation Due to Jitter, Rmin = %5.4f m' % Rmin)
cmin =  (Rmic / (t + block_jitter/fs))
cmax =  (Rmic / (t - block_jitter/fs))
print('NOT meaningful speed of sound changes:')
print('Speed Of Sound Variation Due to Jitter, cmin = %5.4f m/s' % cmin )
print('Speed Of Sound Variation Due to Jitter, cmax = %5.4f m/s' % cmax)
