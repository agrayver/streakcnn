import h5py

import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy import random
from numpy import random

from matplotlib.pyplot import *
import matplotlib.patches as patches

from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageFilter
from PIL.ImageChops import lighter as ImComb

from scipy.ndimage.filters import gaussian_filter
import time
import skimage

from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny

import torch
import math
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Classes
class DNS_StreakImageGenerator:
    def __init__(self, x,y,u,v,densityParticle,radius,CNN_width,CNN_height):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.density = densityParticle
        self.radius = radius
        self.CNN_width=CNN_width
        self.CNN_height=CNN_height
        self.width=self.x.shape[1]
        self.height=self.x.shape[0]

    def generate(self, count = 1, sigma = None, truncate = None):
        images = np.zeros([count, 1, self.height, self.width], dtype=np.float32)
        seedScaling=(self.width//self.CNN_width)*(self.height//self.CNN_height)
        nstreaks = random.randint(seedScaling*self.density.start,\
                                  seedScaling*self.density.stop, count)
        
        for i in range(0, count): 
            imStreaksNoFilter,_xSeed,_ySeed,_DxSeed,_DySeed=\
            generateStreakImageDNS(nstreaks[i],self.radius,\
                                   self.x,self.y,self.u,self.v)    
            if sigma is None:
                images[i,0,:,:] = imStreaksNoFilter;
            else:
                data = np.array(imStreaksNoFilter.getdata(), dtype=float);
                data *= random.uniform(0.1, 10, size=len(data))
                imStreaksNoFilter.putdata(data)
                image = imStreaksNoFilter.filter(ImageFilter.GaussianBlur(radius = sigma))
                data = np.array(image.getdata())
                data[data > 255] = 255
                image.putdata(data)
                images[i,0,:,:] =image
                               
        return images
    
# Functions
def splitImage(processImage,CNN_width,CNN_height,overlap):
    import numpy as np
    
    width=processImage.shape[1]
    height=processImage.shape[0]
    
    Nw=(width-CNN_width)//(CNN_width-overlap)
    Nh=(height-CNN_height)//(CNN_height-overlap)
    _subwindow = np.zeros((Nh*Nw, CNN_width, CNN_height))

    _x=np.zeros(Nh*Nw)
    _y=np.zeros(Nh*Nw)
    indxInterogationWindow=0
    for i in range(Nh):
        for j in range(Nw):
            _subwindow[indxInterogationWindow,:,:]=processImage[\
            (CNN_height-overlap)*i:(CNN_height-overlap)*i+CNN_height,\
            (CNN_width-overlap)*j:(CNN_width-overlap)*j+CNN_width]
            
            _x[indxInterogationWindow]=(CNN_width-overlap)*j+CNN_width//2
            _y[indxInterogationWindow]=(CNN_height-overlap)*i+CNN_height//2
            indxInterogationWindow+=1

    return _x,_y,_subwindow,Nh,Nw

def splitWindows(processImage,CNN_width,CNN_height,overlap):
    
    nImages=processImage.shape[0]
    width=processImage.shape[3]
    height=processImage.shape[2]
    Nw=(width-CNN_width)//(CNN_width-overlap)
    Nh=(height-CNN_height)//(CNN_height-overlap)
    _subwindow = np.zeros((nImages,Nh*Nw, 1, CNN_width, CNN_height))
    _x=np.zeros(Nh*Nw)
    _y=np.zeros(Nh*Nw)
    for iImage in range(nImages):
        indxInterogationWindow=0
        for i in range(Nh):
            for j in range(Nw):
                _subwindow[iImage,indxInterogationWindow,0,:,:]=processImage[iImage,0,\
                    (CNN_height-overlap)*i:(CNN_height-overlap)*i+CNN_height,\
                    (CNN_width-overlap)*j:(CNN_width-overlap)*j+CNN_width]

                _x[indxInterogationWindow]=(CNN_width-overlap)*j+CNN_width//2
                _y[indxInterogationWindow]=(CNN_height-overlap)*i+CNN_height//2
                indxInterogationWindow+=1

    return _x,_y,_subwindow,Nh,Nw


def windowAveraging(displacement,angle,CNN_width,CNN_height,overlap):
    width,height=displacement.shape[1],displacement.shape[0]
    Nw=(width-CNN_width)//(CNN_width-overlap)
    Nh=(height-CNN_height)//(CNN_height-overlap)
    meanDisplacement = np.zeros((Nh,Nw))
    meanAngle = np.zeros((Nh,Nw))
    x=np.zeros((Nh,Nw))
    y=np.zeros((Nh,Nw))

    indxInterogationWindow=0
    for i in range(Nh):
        for j in range(Nw):
            meanDisplacement[i,j]=\
            displacement[(CNN_height-overlap)*i:(CNN_height-overlap)*i+CNN_height,\
                    (CNN_width-overlap)*j:(CNN_width-overlap)*j+CNN_width].mean()
                     
            meanAngle[i,j]=\
            angle[(CNN_height-overlap)*i:(CNN_height-overlap)*i+CNN_height,\
                    (CNN_width-overlap)*j:(CNN_width-overlap)*j+CNN_width].mean()
              
            x[i,j]= overlap*j+CNN_width//2
            y[i,j]= overlap*i+CNN_height//2

 

    return x,y,meanDisplacement,meanAngle, Nh,Nw       

def generateStreakImageDNS(Nparticle,particleRadius,x,y,u,v):
    from PIL import Image
    from PIL import ImageDraw   
    import random
    import numpy as np
    
    width=x.shape[1]
    height=x.shape[0]
    widthLine=particleRadius
    _x0=x.flatten()
    _y0=y.flatten()
    _Dx=u.flatten()
    _Dy=v.flatten()
    indx=np.random.randint(0,_x0.shape[0],size=Nparticle)
    _x1=_x0[indx]+_Dx[indx]
    _y1=_y0[indx]+_Dy[indx]
    
    
    im = Image.new('L', (width,height),(0))
    draw = ImageDraw.Draw(im)
    streaksWidth=np.zeros(Nparticle)
    for i in range(Nparticle):
            streaksWidth = random.randint(particleRadius.start, particleRadius.stop)
            draw.line(((_x0[indx[i]],_y0[indx[i]]),(_x1[i],_y1[i])), \
                      fill=(1000), width=streaksWidth)   

    return im,_x0[indx],_y0[indx],_Dx[indx],_Dy[indx]

def getEnergySpectrum(x1,y1,z):
    '''
    densitySpectrum(x1,y1,z)
    Calculate the 2D density spectrum in space of z
    Input:
    X1,Y1, 1D array with coordinates
    z array (len(y1), len(x1))
    Output:
    kx,FFTx,ky,FFTy,k,FFT,FFT2D
    '''

    dx=np.mean(np.diff(x1))
    dy=-np.mean(np.diff(y1))
    _kx = np.fft.fftfreq(len(x1),dx)
    _ky = np.fft.fftfreq(len(y1),dy)
    _zfft=np.fft.fft2(z)
    _s=np.abs(_zfft)
    Nx = int(len(_kx)/2.)
    Ny = int(len(_ky)/2.)
    kx=_kx[:Nx]
    ky=_ky[:Ny]
    s=_s[0:Ny,0:Nx]

    kmax=np.min([np.max(kx),np.max(ky)])
    kxgrid,kygrid=np.meshgrid(kx,ky,sparse=False)
    _k=np.sqrt(kxgrid**2+kygrid**2)
    dk=np.max([np.mean(np.abs(np.diff(kx))),np.mean(np.abs(np.diff(ky)))])
    
    Nbin=int(kmax/dk)
    ek=np.ones(Nbin)
    k=np.ones(Nbin)
    for i in range(0,Nbin):
            indxk=np.where((_k>i*dk) & (_k<(i+1)*dk))
            ek[i]=np.mean(s[indxk])
            k[i]=(2.*i+1.)*dk/2.
            
    ekx=np.mean(s,axis=0)
    eky=np.mean(s,axis=1)
    return kx,ekx,ky,eky,k,ek,s

def probHoughTransform(image, threshold=15, line_length=4, line_gap=5):
    
    edges = canny(image, 1, 5, 25)
    lines = probabilistic_hough_line(edges, threshold=15, line_length=4,
                                     line_gap=5)
    Dx=[]
    Dy=[]
    for line in lines:
        p0, p1 = line
        dx=p1[0]-p0[0]
        dy=p1[1]-p0[1]
        Dx.append(dx)
        Dy.append(dy)

            
    Dx=np.array(Dx)
    Dy=np.array(Dy)
    displacement=np.sqrt(Dx**2+Dy**2)
    angle=np.rad2deg(np.arctan(Dy/Dx))
    predictionDelta= displacement.mean()
    predictionAngle=angle.mean()
    rmsDelta=displacement.std()
    rmsAngle=angle.std()

    return lines,displacement,angle,predictionDelta,predictionAngle,rmsDelta,rmsAngle