''' This file contains common functions used in the deconvolution GUI routine
     for both CPU and GPU versions'''

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, median_filter
from scipy.misc import imresize
import pywt

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

def remove_blob_sino(sinogram, sigma, thresh):
  
    sinom = median_filter(sinogram, size=int(sigma))
    sino0 = np.copy(sinogram)
    
    # Get the histogram of the original sinogram
    b1 = np.arange(np.min(sinogram),np.max(sinogram),2048)
    h1, bins1 = np.histogram(sinogram, bins = b1)

    # Get the histogram of the filtered image
    h2, bins2 = np.histogram(sinom, bins = b1)
    # plt.bar(bins2, h2)
    # plt.show()

    # intensity value at which the median filtered image has no pixels
    # it is our threshold
    th = b1[np.sum(h2[:int(thresh)]==0)]

    # Creating masks based on the threshold
    mask = sinogram<th
    
    sinogram1 = np.flipud(np.roll(sinogram, sinogram.shape[1]//2, axis=1))
    
    sino0[sinogram<th] = sinogram1[sinogram<th]

    return (mask, sino0)

def std_mask(array, x):
    
    # Define a mask based on intensity outside the boundaries 
    # [-x*std, +x*std] with x passed as parameter 
    return (array < np.mean(array)-x*np.std(array)) | (array > np.mean(array)+x*np.std(array))


def remove_blob_sino_wavelet(sinogram, sigma):
    
    # define the wavelet type
    wl = 'haar'

    # First order wavelet decomposition
    coeffs = pywt.dwt2(sinogram, wl)

    # Extract coefficients
    cA, (cH, cV, cD) = coeffs

    
    # Median filter the coefficients of the decomposition
    cAm = median_filter(cA, size=sigma)
    cHm = median_filter(cH, size=int(2*sigma))
    cVm = median_filter(cV, size=int(2*sigma))
    cDm = median_filter(cD, size=int(2*sigma))

    # Filter the coefficients. For points in the mask replace the value with the value obtained 
    # from the median filtered versions of the same arrays
    xt = 3
    cA[std_mask(cA, x=2)==1] = cAm[std_mask(cA, x=2)==1]
    cH[std_mask(cH, x=xt)==1] = cHm[std_mask(cH, x=xt)==1]
    cV[std_mask(cV, x=xt)==1] = cVm[std_mask(cV, x=xt)==1]
    cD[std_mask(cD, x=xt)==1] = cDm[std_mask(cD, x=xt)==1]

    
    # Inverse wavelet decomposition. Return the sinogram after filtering
    # the wavelet coefficients
    sinow = pywt.idwt2(coeffs, wl)
    
    if sinogram.shape[1]/2.0 != sinogram.shape[1]//2:
        sinow = sinow[:,:-1]
    #sinow = imresize(sinow, (sinogram.shape[0], sinogram.shape[1]), interp='bilinear')

    return sinow.astype('float32')