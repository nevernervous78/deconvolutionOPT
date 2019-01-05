''' This file contains functions used in the deconvolution GUI routine
    using only standard Numpy functions. Compatible with non-NVIDIA machines'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from common_utilities import corrCoeff, sino_centering, psf1d_data

def corrCoeff(arr1, arr2):

    return np.correlate(arr1, arr2)[0] \
           / np.sqrt(np.correlate(arr1, arr1))[0] \
           / np.sqrt(np.correlate(arr2, arr2))[0]

def sino_centering(sinogram, span=50, fullrot=True):

    ''' Automated misalignment compensation for OPT scans'''  

    nx,nangles = sinogram.shape

    coeff = np.zeros(span)
    for i in range(span):

        sinos = np.roll(sinogram, (i-span//2), axis=0)
        if fullrot is True:
            coeff[i] = corrCoeff(sinos[:,0], np.flipud(sinos[:,nangles//2]))
        else:
            coeff[i] = corrCoeff(sinos[:,0], np.flipud(sinos[:,-1]))

    return np.argmax(coeff)-span//2

def psf1d_data(mask, shape):
    # This is already in Fourier space
    return np.outer(gaussian_filter1d(mask.astype('float32'), 100), np.ones(shape[1]))


def runDeconvolutionCPU(sinogram, pixel_size, noise_level=0.05):

    # Subtract the mean from the sinogram
    sinogram -= -np.mean(sinogram)

    # Calculate the FT of the sinogram 
    fsino = np.fft.fft2(sinogram)
    #fsino = fft2_gpu(sino)
    fsinot = np.fft.fft2(sinogram)

    # Get the power spectrum in the vertical direction (pixel axis)
    vertps = np.mean(np.abs(np.fft.fftshift(fsino)), axis=1)

    # Get the baseline (first 200 pixels) corresponding to noise
    baseline = np.mean(vertps[:200])

    # Mask out everything that is smaller than the baseline + 50% after Gaussian smoothing with sigma=3
    mask = gaussian_filter1d(vertps,3) > 1.4*baseline 

    # Generate psf from data
    psf_d = np.fft.fftshift(psf1d_data(mask, sinogram.shape))

    # Define the angular range in rad (360 deg rotation)        
    Phi = np.linspace(-sinogram.shape[1]/(4*np.pi), sinogram.shape[1]/(4*np.pi), sinogram.shape[1], True) 

    # Define the transverse horizontal coordinate in Fourier space       
    Rx = np.linspace(-0.5/pixel_size, 0.5/pixel_size, sinogram.shape[0], True)

    line = np.outer(1.0/Rx, Phi)

    # Define the roll-off filter
    Wr = np.zeros_like(fsino)
    w = 0.3
    for i in range(fsino.shape[0]):
        for j in range(fsino.shape[1]):
      
            if line[i,j] <= 0.:
                Wr[i,j] = 1.0
            elif line[i,j] > w:
                Wr[i,j] = 0.000001
            else:
                Wr[i,j] = np.cos( (np.pi/2)*np.abs(line[i,j])/w)

    # Roll-off filter combined to Wiener filter
    fsino_dec = Wr*fsino*np.conj(psf_d)/(psf_d*np.conj(psf_d)+noise_level)

    sino_dec = np.real(np.fft.ifft2(fsino_dec) )

    return sino_dec

