import keras
import matplotlib as mpl  # colorbar handling
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tools
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # do not use GPU

# deactivate Tensorflow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model_file = "tmp/models/sfa_cnn.h5"

custom_obj_dict = {'SplitLayer': tools.layers.SplitLayer,
                   'Corr1D': tools.layers.Corr1D,
                   'jensen_shannon_divergence': tools.losses.jensen_shannon_divergence
                   }

model = keras.models.load_model(model_file, custom_objects=custom_obj_dict)
model.summary()

train_path = "tmp/train/"
dirsound = np.load(train_path + "anechoic_rigid_sphere_noise_E0_signals.npy")
dirlabels = np.load(train_path + "anechoic_rigid_sphere_noise_E0_azimuths.npy")
print(dirsound.shape, dirlabels.shape)

print(dirlabels)
Ndirs = 360
phis = np.arange(0, Ndirs) - 180  # -180 due to SOFA file handling

Nt = 64  # hard coded, since this is the resolution of the model
Nmic = dirsound.shape[1]
Ntraining = dirsound.shape[2]

Nshift = 32

print('Mic:', Nmic)
delta_az = 360. / Nmic
print('delta az = %3.1f deg' % delta_az)
print('delta az/2 = %3.1f deg' % (delta_az/2))
print('delta az/4 = %3.1f deg' % (delta_az/4))



print('### EVAL 1###')
filename = 'tmp/models/prediction_accuracy_freefield_single_PDF'

# dB, impulse response peak to impulse response noise floor ratio, aka PNR
PNR = np.array([30,60,90])
az = np.arange(0,360,45)

for pnr_check in PNR:
    plt.figure(figsize=(10,2.5))
    for az_check in az:
        for n in range(az_check, Ntraining, Ndirs):

            # set up +- samples shift around peak in the middle of array (from lin phase lowpass sinc)
            shift = np.random.randint(low=-Nshift, high=+Nshift, size=1, dtype='l') 
            # apply shift as cyclic shift, this is ok here since IRs decay rapidly
            irs = np.roll(dirsound[Nt-32:Nt+32,:,n],shift,axis=0)

            # get rms of noise for desired PNR
            sigma = np.sqrt( np.amax(irs) / 10**(pnr_check/10))
            # set up gaussian white noise with rms
            noise = sigma * np.random.randn(Nt, Nmic)

            # apply noise and set up for model prediction
            feature_test = np.expand_dims(irs + noise, axis=0)
            label_test = dirlabels[n]

            # predict
            y_predict = np.squeeze(model.predict(feature_test))
            doa_predict = phis[np.argmax(y_predict)]
            #print('true, predicted: %+3.1f | %+3.1f deg' % (label_test, doa_predict))

            cl = np.squeeze(np.random.randint(low=0, high=255,size=(3,1))/255.)
            plt.plot(phis, y_predict, color=cl)
            plt.plot([label_test, label_test], [np.max(y_predict), np.max(y_predict)], 'o', ms=6, color=cl)

    plt.xticks(np.arange(-180, 180 + 45, 45))
    plt.xlabel('azimuth in deg')
    plt.ylabel('p(DOA)')
    plt.grid(True)
    plt.title('dot: true, pdf: predicted, PNR='+str(pnr_check)+'dB')
    plt.savefig(filename+'_PNR'+str(pnr_check)+'dB_Nshift'+str(Nshift)+'_allR.png', dpi=300)


print('### EVAL 2 ###')
filename = 'tmp/models/prediction_accuracy_freefield_all_dir'

dirlabels = np.load(train_path + "anechoic_rigid_sphere_noise_E0_azimuths.npy")
Nr = 10  # radius condition
dirlabels = dirlabels[0+Nr*360:360+Nr*360]
N = dirlabels.shape[0]

#impulse response peak to impulse response noise floor ratio  
PNR = np.array([30,60,90])

for pnr_check in PNR:
    print(pnr_check)
    doa_predict = np.zeros(N)
    for n in range (0,N):
        # set up +- samples shift around peak in the middle of array (from lin phase lowpass sinc)
        shift = np.random.randint(low=-Nshift, high=+Nshift, size=1, dtype='l') 
        # apply shift as cyclic shift, this is ok here since IRs decay rapidly
        irs = np.roll(dirsound[Nt-32:Nt+32,:,n],shift,axis=0)  # 64er length hard coded!!!  

        sigma = np.sqrt( np.amax(irs) / 10**(pnr_check/10))
        noise = sigma * np.random.randn(Nt, Nmic)
        #print('PNR check:', 10*np.log10(np.amax(irs) / np.std(noise, axis=0)**2), 'dB')
        y_predict = np.squeeze(model.predict(np.expand_dims(irs + noise, axis=0)))
        doa_predict[n] = phis[np.argmax(y_predict)]

    doa_predict360 =  np.copy(doa_predict)
    doa_predict180 =  np.copy(doa_predict)
    dirlabels360 =  np.copy(dirlabels)
    dirlabels180 =  np.copy(dirlabels)  
    # wrap to show 0...360 deg:   
    doa_predict360[doa_predict360<0] += 360
    dirlabels360[dirlabels360<0] += 360
    print(dirlabels180.shape)

    fig, ax = plt.subplots(2,2)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    ax[0,0].plot(dirlabels180, 'C2', label='true', lw=3)
    ax[0,0].plot(doa_predict180, 'C3', label='predicted', lw=2)
    ax[1,0].plot(dirlabels180-doa_predict180, 'C0', lw=2, label='error = true - predicted')

    ax[0,1].plot(dirlabels360, 'C2', label='true', lw=3)
    ax[0,1].plot(doa_predict360, 'C3', label='predicted', lw=2)
    ax[1,1].plot(dirlabels360-doa_predict360, 'C0', lw=2, label='error = true - predicted')

    for axi in ax.flatten():
        axi.set_xticks(np.arange(0,360,45))
        axi.set_xlim(0, 360)
        axi.grid(True)
        axi.legend()
        axi.set_xlabel('condition, here ident with azimuth DOA')
        axi.set_ylabel('deg')

    ax[0,0].set_yticks(np.arange(-180,180+30,30))
    ax[0,0].set_ylim(-180,180)
    ax[0,1].set_yticks(np.arange(0,360+30,30))
    ax[0,1].set_ylim(0,360)
    for axi in ax[1,:].flatten():
        axi.set_yticks(np.arange(-30,30+5,5))
        axi.set_ylim(-20,20)
        
    Ns = 1  # not in for loop, but to make filenames consistent
    plt.savefig(filename+'_SingleNs'+str(Ns)+'_SingleNr'+str(Nr)+'_PNR'+str(pnr_check)+'dB_Nshift'+str(Nshift)+'.png', dpi=300)



print('### EVAL 3 ###')
dirlabels = np.load(train_path + "anechoic_rigid_sphere_noise_E0_azimuths.npy")

Ndirs = 360
data = np.zeros((Ndirs,Ndirs))

PNRmin = 30  # dB
PNRmax = 66  # dB
PNRdelta = 6  # dB
PNRfreq = 10  # multiple check per PNR value to get a PDF like impression of the prediction performance
# since we don't care on the exact PNR value order we can
# either
#PNR = np.repeat(np.arange(PNRmin, PNRmax, PNRdelta), PNRfreq)
# or, 
PNR = np.tile(np.arange(PNRmin, PNRmax, PNRdelta), PNRfreq) 

# number of considered wavefront curvatures, i.e. point source distance to array
Nr = Ntraining//Ndirs
# number of randomized time shifts
Ns = 10

# debug values
#Nr = 2
#Ns = 2

print(Ns,Nr,PNR.shape,Nshift)
print(PNR)


for n in range(Ndirs):  # do for all prediction directions
    if np.mod(n,10)==0:
        print(n)
    # we have a new direction, so clear the 'summed probability'-variable:
    y_predict_sum = np.zeros(Ndirs)  
    for r in range(Nr):  # do for all wavefront curvatures/_r_adii stored in data set
        #print('r:', r)
        for s in range(Ns):  # check multiple time _s_hifts
            #print('s:', s)
            # set up +- samples shift around peak in the middle of array (from lin phase lowpass sinc)
            shift = np.random.randint(low=-Nshift, high=+Nshift, size=1, dtype='l') 
            # get free field responses for intended directions and intended wavefront curvature
            #print('current direction:', dirlabels[n + r*Ndirs])
            # apply shift as cyclic shift, this is ok here since IRs decay rapidly
            irs = np.roll(dirsound[Nt-32:Nt+32,:,n + r*Ndirs], shift, axis=0)  # 64er length hard coded!!!      

            # do for all desired PNRs
            for m, val in enumerate(PNR):  
                sigma = np.sqrt( np.amax(irs) / 10**(val/10))  # get noise rms
                noise = sigma * np.random.randn(Nt, Nmic)  # generate noise with desired rms
                feature_test = np.expand_dims(irs + noise, axis=0)  # add noise and prep shape
                y_predict = np.squeeze(model.predict(feature_test))  # predict
                y_predict_sum += y_predict  # sum probabilities because we consider p(cond1) or p(cond2) or ...
    data[:,n] = y_predict_sum  # store summed probability for the current direction and proceed with next direction
    #print('### next direction ###')
y_predict.shape


filename = 'tmp/models/prediction_accuracy_freefield'

Ncol = 101  # number of colors in colorbar

data /= np.max(data)
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(1, 1, 1)
col_tick = np.linspace(0, np.max(data), Ncol, endpoint=True)
cmap = mpl.cm.get_cmap('magma_r')
norm = mpl.colors.BoundaryNorm(col_tick, cmap.N)
surf = ax.imshow(data, cmap=cmap, norm=norm, extent=[0, 360, 180, -180], aspect='equal')  # pay attention to xy ticks!!!
cbar = fig.colorbar(surf, ax=ax, ticks=col_tick[::Ncol//5], label=r'p$_\Sigma$ norm to max of 1')

ax.plot([0,180], [0,180], 'C7', ls='-.')
ax.plot([180,360], [-180,0], 'C7', ls='-.')

offs = 10
ax.plot([-offs,180-offs], [0,180], 'C7', ls='--', lw=0.75)
ax.plot([+offs,180+offs], [0,180], 'C7', ls='--', lw=0.75)
ax.plot([180+offs,360+offs], [-180,0], 'C7', ls='--', lw=0.75)
ax.plot([180-offs,360-offs], [-180,0], 'C7', ls='--', lw=0.75)

offs = 20
ax.plot([-offs,180-offs], [0,180], 'C7', ls=':', lw=0.75)
ax.plot([+offs,180+offs], [0,180], 'C7', ls=':', lw=0.75)
ax.plot([180+offs,360+offs], [-180,0], 'C7', ls=':', lw=0.75)
ax.plot([180-offs,360-offs], [-180,0], 'C7', ls=':', lw=0.75)

ax.set_xlim(0, 360)
ax.set_ylim(-180, +180) # pay attention to xy ticks!!!
ax.set_xticks(np.arange(0,360+45,45))
ax.set_yticks(np.arange(-180,180+45,45))
ax.set_xlabel('true DOA in deg')
ax.set_ylabel('predicted DOA in deg')

ax.text(5,-130, r'-.-.- $\rightarrow\quad 0$ deg', color='C7')
ax.text(5,-155, r'----- $\rightarrow\pm 10$ deg', color='C7')
ax.text(5,-170, r'..... $\rightarrow\pm 20$ deg', color='C7')

ax.grid(True)

plt.savefig(filename+'_Ns'+str(Ns)+'_Nr'+str(Nr)+'_PNR'+str(len(PNR))+'_Nshift'+str(Nshift)+'.png', dpi=300)