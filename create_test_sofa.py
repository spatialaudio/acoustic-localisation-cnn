from time import strftime
import numpy as np
import os
import sofa

# AES69-2015 Standard for Spatial Acoustic Data File Format
# print(sofa.conventions.implemented())

# for our case we 'abuse' the intended source/emitter handling
# we rather set up several individual point sources with
# absolute coordinates (x,y,z) as emitters and completely ignore the info of
# the (must be single!) source
# thus, we cover the acoustic transfer paths:
# multiple point sources to multiple mic array positions
# including their room reflections
S = 1  # source, must be 1, thus S is not declared with an explicit variable
# S=1 as must have is a flaw in the standard :-(
E = 20  # the source has E emitters, e.g. an IKO loudspeaker has E=20 speakers
M = 8  # measurements, here: number of mic array positions within the room
R = 32  # the mic array has R receivers, e.g. an EigenMike has R=32 microphones

C = 3  # fixed, must be 3, geometry coordinate dimension
I = 1  # fixed, must be 1, singleton dimension

N = 512  # samples in time domain for FIR
fs = 441000.  # Hz, sampling frequency of the data set

# required extra metadata for the DNN validation phase:
c = 343.  # m/s, speed of sound
# in future work we should use RoomTemperature to cover this
# however since we deal with simulations where c=const, this is ok here
SNR = 20.  # dB, peak to noise ratio
room_x = 10.  # m, room width
room_y = 11.  # m, room height
room_z = np.sqrt(room_x * room_y)  # m, room height
# HorizontalImageSourceVariables -> 'no, azi, colat, distance, time'
# e.g.
image_src = np.array([[1, 49.88, 90.00, 4.29, 12.87],
                      [2, -72.40, 90.00, 9.15, 27.44],
                      [3, 16.95, 90.00, 11.25, 33.76],
                      [4, 163.72, 90.00, 11.70, 35.11],
                      [5, -39.00, 90.00, 13.85, 41.56],
                      [6, -142.19, 90.00, 14.22, 42.66],
                      [7, 80.91, 90.00, 17.50, 52.50],
                      [8, 170.32, 90.00, 19.51, 58.54],
                      [9, 58.08, 90.00, 20.36, 61.08],
                      [10, 123.03, 90.00, 20.61, 61.84],
                      [11, -83.06, 90.00, 22.89, 68.66]])
N_image_src = image_src.shape[0]

# absolute position as cartesian, actually don't care
SourcePosition = np.zeros((I, C))
print('SourcePosition', SourcePosition.shape)

# we abuse this for absolute positions as cartesian
EmitterPosition = np.random.rand(E, C)
print('EmitterPosition', EmitterPosition.shape)

# absolute positions as cartesian
ListenerPosition = np.random.rand(M, C)
print('ListenerPosition', ListenerPosition.shape)

# relative positions as cartesian
ReceiverPosition = np.random.rand(R, C)
print('ReceiverPosition', ReceiverPosition.shape)

irs = np.zeros((M, R, E, N))  # that's how Data.IR gets stored in the SOFA file
# so could already provide this dimension order
print('irs', irs.shape)

sofafile = 'tmp/test.sofa'
# os.remove(sofafile)

DRIR = sofa.Database.create(sofafile, "GeneralFIRE", M)
DRIR.Metadata.set_attribute('AuthorContact', 'Schultz/Winter/Spors')
DRIR.Metadata.set_attribute('Organization', 'INT/IEF/Uni Rostock')
ts = 'Test-Room for DOA Identification of Early Reflections' + \
     '(DOA-IEAR) with DNN for DAGA 2020'
DRIR.Metadata.set_attribute('Title', ts)
DRIR.Metadata.set_attribute('Comment', 'WIP...')
DRIR.Metadata.set_attribute('DatabaseName', 'daga2020_doa_iear_dnn')

DRIR.Metadata.set_attribute('ListenerDescription',
                            'Circ hor mic array, coordinates are absolute')
DRIR.Metadata.set_attribute('ReceiverDescription',
                            'Ideal 3D monopole receiver, coordinates are relative')

DRIR.Metadata.set_attribute('SourceDescription', 'Not Used')
DRIR.Metadata.set_attribute('EmitterDescription',
                            'Ideal 3D monopole transmitter, coordinates are absolute')

DRIR.Metadata.set_attribute('RoomType', 'reverberant')
DRIR.Metadata.set_attribute('RoomDescription', 'Test Room #')

DRIR.Metadata.set_attribute('License', 'CCBY4.0')
DRIR.Metadata.set_attribute('DateCreated', strftime("%Y-%m-%d %H:%M:%S"))
DRIR.Metadata.set_attribute('DateModified', 'not modified')

DRIR.Source.initialize(sofa.spatial.Set(sofa.spatial.Coordinates.State.Fixed))
DRIR.Source.Position.set_system(ctype='cartesian', cunits='m, m, m')
DRIR.Source.Position.set_values(SourcePosition, dim_order=('I', 'C'))

DRIR.Emitter.initialize(sofa.spatial.Set(sofa.spatial.Coordinates.State.Fixed),
                        count=E)
DRIR.Emitter.Position.set_system(ctype='cartesian', cunits='m, m, m')
DRIR.Emitter.Position.set_values(EmitterPosition, dim_order=('E', 'C'),
                                 repeat_dim=('I'))

DRIR.Listener.initialize(
    sofa.spatial.Set(sofa.spatial.Coordinates.State.Varying))
DRIR.Listener.Position.set_system(ctype='cartesian', cunits='m, m, m')
DRIR.Listener.Position.set_values(ListenerPosition, dim_order=('M', 'C'))

DRIR.Receiver.initialize(
    sofa.spatial.Set(sofa.spatial.Coordinates.State.Fixed), count=R)
DRIR.Receiver.Position.set_system(ctype='cartesian', cunits='m, m, m')
DRIR.Receiver.Position.set_values(ReceiverPosition, dim_order=('R', 'C'),
                                  indices={'I': 0, 'M': 0})

DRIR.Data.initialize(N, False)
DRIR.Data.SamplingRate.set_value(fs)
DRIR.Data.IR.set_values(irs, dim_order=('M', 'R', 'E', 'N'))

DRIR.Metadata.set_attribute('DAGA_SpeedOfSound', c)
DRIR.Metadata.set_attribute('DAGA_SpeedOfSoundUnit', 'm/s')
DRIR.Metadata.set_attribute('DAGA_PeakToNoiseRatio', SNR)
DRIR.Metadata.set_attribute('DAGA_PeakToNoiseRatioUnit', 'dB')

DRIR.Metadata.set_attribute('DAGA_RoomDimensionx', room_x)
DRIR.Metadata.set_attribute('DAGA_RoomDimensionxUnit', 'm')
DRIR.Metadata.set_attribute('DAGA_RoomDimensiony', room_y)
DRIR.Metadata.set_attribute('DAGA_RoomDimensionyUnit', 'm')
DRIR.Metadata.set_attribute('DAGA_RoomDimensionz', room_z)
DRIR.Metadata.set_attribute('DAGA_RoomDimensionzUnit', 'm')

DRIR.Metadata.set_attribute('DAGA_HorizontalNumberOfImageSources',N_image_src)
DRIR.Metadata.set_attribute('DAGA_HorizontalImageSourceVariables',
                            'no, azi, colat, distance, time')
DRIR.Metadata.set_attribute('DAGA_HorizontalImageSourceUnits',
                            '-, deg, deg, m, ms')

for e in range(E):
    for m in range(M):
        for image_src_idx in range(N_image_src):
            DRIR.Metadata.set_attribute('DAGA_HorizontalImageSource_' + str(e) + '_' + str(m) + '_' + str(image_src_idx+1), image_src[image_src_idx,:])

print()

DRIR.Metadata.dump()
DRIR.Dimensions.dump()
DRIR.close()
