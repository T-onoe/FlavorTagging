#=======================================================================================================#
#===TOOLS FOR DATA ANALYSIS=============================================================================#
#=======================================================================================================#
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def file_open(file):

     import sys
     datas = []
     try:
         f = open(file, 'r', encoding = 'utf-8')
     except Exception:
         print("file open error. file not found:", str(file))
         sys.exit(1)
     for line in f:
#        line = line.strip()
#        line = line.replace('\n','')
#        line = line.split(",")
         lines = line[:-1].split()
         datas.append(lines)
     f.close()
     print("file open !!")

     return datas


def shaper_tanh(x, a=1.0, b=1.0, c=0.0, d=0.0):

    return a*np.tanh(b*(x-c))+d


def shaper_linear(x, a=1.0, b=0.0, c=0.0):

    return a*(x-b)+c


def logarithm(x):

     return np.log10(x + 1.0)


def abs(x):
     return abs(x)


def logarithm1(x):
     
     return np.log10(x/(10E-2) + 1.0)


def logarithm2(x):
     return np.log10(x/(10E-3) + 1.0)


def logarithm3(x):
     return np.log10(x/(10E-4) + 1.0)


def logarithm4(x):
     return np.log10(x/(10E-5) + 1.0)

def logarithm5(x):
     return np.log10(x/(5*10E-4) + 1.0)

def logarithm6(x):
     return np.log10(x/(3*10E-4) + 1.0)

def logarithm7(x):
     return np.log10(x/(5*10E-3) + 1.0)


def rescaling(x):

    min = np.amin(x)
    max = np.amax(x)

    return (x - min) / (max - min)

def sqrt(x):

     return np.sqrt(x)

def standardization(x):

     mean = np.mean(x)
     std = np.std(x)

     if std == 0:
          return x - mean
     else:
          return (x - mean) / std

def under_sample(X, Y):

    from imblearn.under_sampling import RandomUnderSampler

    numy = np.min(np.count_nonzero(Y==0), np.count_nonzero(Y==1), np.count_nonzero(Y==2), np.count_nonzero(Y==3), np.count_nonzero(Y==4))
    rus = RandomUnderSampler(sampling_strategy={0:numy, 1:numy, 2:numy, 3:numy, 4:numy}, random_state=0) 
    X_resampled, Y_resampled = rus.fit_sample(X, Y)

    return X_resampled, Y_resampled


def cartesian2polar(position):
    
    num = position.shape[0]
    newPosition = np.empty([num,3], dtype=np.float64)
    newPosition[:,0] = np.linalg.norm(position, axis=1)
    newPosition[:,1] = np.arccos(position[:,2] / newPosition[:,0])
    newPosition[:,2] = np.arctan2(position[:,1], position[:,0])
    nan_index = np.isnan(newPosition[:,1])
    newPosition[nan_index,1] = 0

    return newPosition


def tracker(t, pi, d0, z0, phi, omega, tanlam, charge, r):

    x = -d0 * np.sin(phi) + r * np.cos(phi - pi/2 * charge) + r * np.cos(-charge * t + phi + pi/2 * charge)
    y =  d0 * np.cos(phi) + r * np.sin(phi - pi/2 * charge) + r * np.sin(-charge * t + phi + pi/2 * charge)
    z =  z0 + r * t * tanlam

    return x, y, z


def t_tracker(tr1, tr2, t, curvature=False):

    tr1d0, tr1z0, tr1phi, tr1omega, tr1tanlam, tr1charge = tr1[0], tr1[1], tr1[2], tr1[3], tr1[4], tr1[5]
    tr2d0, tr2z0, tr2phi, tr2omega, tr2tanlam, tr2charge = tr2[0], tr2[1], tr2[2], tr2[3], tr2[4], tr2[5]
    pi = np.pi

    tr1r = 1/tr1omega
    tr2r = 1/tr2omega
    
    if curvature:
        tr1t = t/tr1r
        tr2t = t/tr2r
    else:
        tr1t = t
        tr2t = t
        
    track1 = []
    track2 = []

    for tr1dt, tr2dt in zip(tr1t, tr2t):
        tr1x, tr1y, tr1z = tracker(tr1dt, pi, tr1d0, tr1z0, tr1phi, tr1omega, tr1tanlam, tr1charge, tr1r)
        tr2x, tr2y, tr2z = tracker(tr2dt, pi, tr2d0, tr2z0, tr2phi, tr2omega, tr2tanlam, tr2charge, tr2r)
        track1.append([tr1x, tr1y, tr1z])
        track2.append([tr2x, tr2y, tr2z])

    track1, track2 = np.array(track1, dtype='float32'), np.array(track2, dtype='float32')

    return track1, track2


def plot_tracker(x, t):

    tr1d0, tr1z0, tr1phi, tr1omega, tr1tanlam, tr1charge = x[0], x[1], x[2], x[3], x[4], x[5]
    tr2d0, tr2z0, tr2phi, tr2omega, tr2tanlam, tr2charge = x[21], x[22], x[23], x[24], x[25], x[26]
    pi = np.pi

    tr1r = 1/tr1omega
    tr2r = 1/tr2omega
    
    track1 = []
    track2 = []
    for dt in t:
        tr1x, tr1y, tr1z = tracker(dt, pi, tr1d0, tr1z0, tr1phi, tr1omega, tr1tanlam, tr1charge, tr1r)
        tr2x, tr2y, tr2z = tracker(dt, pi, tr2d0, tr2z0, tr2phi, tr2omega, tr2tanlam, tr2charge, tr2r)
        track1.append([tr1x, tr1y, tr1z])
        track2.append([tr2x, tr2y, tr2z])

    track1, track2 = np.array(track1, dtype=float), np.array(track2, dtype=float)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot(track1[:, 0], track1[:, 1], track1[:, 2], marker=".")
    ax.plot(track2[:, 0], track2[:, 1], track2[:, 2], marker=".")

    plt.show()

def get_symmetric(a, use_tril=True):
    if use_tril:
        a = np.tril(a)
    else:
        a = np.triu(a)
    return a + a.T - np.diag(a.diagonal())
