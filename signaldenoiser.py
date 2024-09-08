# Crear una señal pura
# Añadir ruido
# Aplicar algoritmo de de-noising
# Comparar la señal filtrada con la señal original

# Smoothing running-mean filter: yt = (2k + 1)^-1 ∑ (desde i=t-k, donde t+k) xi // k es un valor, t es la señal en el tiempo

import numpy as np
from scipy.io.wavfile import write 
from scipy.signal import detrend 
import matplotlib.pyplot as plt
import copy

N = 10001
time = np.linspace(0, 4 * np.pi, N)

sampling_rate = 44100
duration = 20.0
frequency = 440.0
t = np.linspace(0.0, duration, int(sampling_rate * duration), endpoint=False)

signal = np.zeros(N)

for j in range(1, 4):
    signal += np.cos (j * time)**j

noisysignal = signal + np.random.randn(N)

plt.plot(time, noisysignal, time, signal)
plt.show()

# Implementar el running-mean filter

filtsignal = noisysignal
k = 15

for t in range(N):
    lowbound = np.max((0, t-k))
    uppbound = np.min((N, t+k))
    filtsignal[t] = np.mean(noisysignal[lowbound : uppbound])

plt.plot(time, filtsignal, time, noisysignal + 0.5)
plt.show()

# Definimos el running-mean filter para usar más adelante nuevamnte

def meansmooth(signalIn, k):
    filtsignal = copy.deepcopy(signalIn)
    for t in range(N):
        filtsignal[t] = np.mean(noisysignal[np.max ((0, t-k)) : np.min((N, t+k)) ])
    return filtsignal

kvals = np.arange(5, 41)
signalCorrs = []

for ki in kvals:
    # Filtar la señal
    fsig = meansmooth(noisysignal, ki)
    # Relación entre la señal filtrada y la original
    signalCorrs.append( np.corrcoef(fsig, signal)[0, 1] )

# Correlación de señal entre la original y la procesada
# print(signalCorrs)

plt.plot(kvals, signalCorrs, 'ks-')
plt.show()

signal = signal.astype(np.int16)
noisysignal = noisysignal.astype(np.int16)
filtsignal = filtsignal.astype(np.int16)

# write("Signal_Mean.wav", sampling_rate, signal)
# write("Noisy_Signal_Mean.wav", sampling_rate, noisysignal)
# write("Filtered_Signal_Mean.wav", sampling_rate, filtsignal)

# Método Número 2 para limpiar una señal
# Gaussian convolution
# Crear señal

srate = 512
time = np.arange(-2, 2+1 / srate, 1 / srate)
pnts = len(time)

signal = detrend (time ** 3 + np.sign(time))
noisysignal = signal + ( np.random.randn(pnts) * 1.1 )


plt.plot(time, noisysignal, time, signal)


# Crear señal Guasiana
k = 10
x = np. arange (-k, k) / srate
s = .005
gkern = np.exp( -x**2 / (2 * s**2) )

plt.plot(x, gkern, 's-')
plt.title('n=%s, s=%g'%(2*k+1, s))
plt.show()

gkern /= sum(gkern)
filtsig = np.convolve(noisysignal, gkern, mode = 'same')
plt.plot(time, noisysignal, time, filtsig, time, signal)
plt.legend(['noisy sig', 'filt sig', 'original'])
plt.xlim(time[[0, -1]])
plt.show()

# Definir rangos de parámetros
krange = np.arange(3, 303, 20)
srange = np.linspace(0.001, 0.5, 60)

# Inicializamos los datos
sseMat = np.zeros( (len(krange), len(srange)) )

allKernels = [ [0] * len(srange) for i in range(len(krange)) ]

for ki in range(len(krange)):
    for si in range(len(srange)):
        # Crear Gauss
        x = np. arange ( -krange[ki], krange[ki]+1 ) / srate
        s = .005
        gkern = np.exp( -x**2 / (2 * srange[si]**2) )
        # Filtrar Señal (Convolución)
        filtsig = np.convolve(noisysignal, gkern/sum(gkern) , mode = 'same')
        #Computar SSE
        sseMat[ki, si] = np.sum((filtsig-signal)**2)
        allKernels[ki][si] = gkern

plt.imshow(sseMat, vmax = 400 ,extent=[srange[0], srange[-1], krange[-1], krange[0]])
plt.gca().set_aspect(1/plt.gca().get_data_ratio())
plt.colorbar()
plt.show()

# ESTA SECCION MUESTRA VARIOS DE LOS KERNELS USADOS EN LA CONVOLUCION DE LA SEÑAL

fig, ax = plt.subplots(4, 4, figsize=(10, 8))

sidx = np.linspace(0, len(srange)-1, 4).astype(int)
kidx = np.linspace(0, len(krange)-1, 4).astype(int)

for si in range(4):
  for kj in range(4):
    ax[kj, si].plot(allKernels [kidx[kj] ][sidx[si]] )
    ax[kj, si].set_xticks([])
    ax[kj, si].set_ylim([0, 1.1])
    ax[kj, si].set_title(r'k=%g, $\sigma$=%.2f' % (krange[kidx[kj]], srange[sidx[si]]))
    ax[kj, si].set_aspect(1/ax[kj, si].get_data_ratio())

plt.show()