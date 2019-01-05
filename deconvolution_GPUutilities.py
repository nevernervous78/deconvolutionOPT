''' This file contains functions used in the deconvolution GUI routine
    using GPU'''

import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import skcuda.fft as cu_fft
from scipy.ndimage.filters import gaussian_filter1d
from common_utilities import corrCoeff, sino_centering, psf1d_data


def fft2_gpu(x, fftshift=False):
    
    ''' This function produce an output that is 
    completely compatible with numpy.fft.fft2
    The input x is a 2D numpy array'''

    # Convert the input array to single precision float
    if x.dtype != 'float32':
        x = x.astype('float32')

    # Get the shape of the initial numpy array
    n1, n2 = x.shape
    
    # From numpy array to GPUarray
    xgpu = gpuarray.to_gpu(x)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    y = gpuarray.empty((n1,n2//2 + 1), np.complex64)
    
    # Forward FFT
    plan_forward = cu_fft.Plan((n1, n2), np.float32, np.complex64)
    cu_fft.fft(xgpu, y, plan_forward)
    
    left = y.get()

    # To make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version
    # We must take care of handling even or odd sized array to get the correct 
    # size of the final array   
    if n2//2 == n2/2.0:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,1:-1],1,axis=0)
    else:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,:-1],1,axis=0) 
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = np.hstack( (left,right) )
    else:
        yout = np.fft.fftshift(np.hstack((left,right)))

    return yout.astype('complex128')

def ifft2_gpu(y, fftshift=False):

    ''' This function produce an output that is 
    completely compatible with numpy.fft.ifft2
    The input y is a 2D complex numpy array'''

    # Convert the input array to complex64
    if y.dtype != 'complex64':
        y = y.astype('complex64')

    # Get the shape of the initial numpy array
    n1, n2 = y.shape
    
    # From numpy array to GPUarray. Take only the first n2/2+1 non redundant FFT coefficients
    if fftshift is False:
        y2 = np.asarray(y[:,0:n2//2 + 1], np.complex64)
    else:
        y2 = np.asarray(np.fft.ifftshift(y)[:,:n2//2+1], np.complex64)
    ygpu = gpuarray.to_gpu(y2) 
     
    # Initialise empty output GPUarray 
    x = gpuarray.empty((n1,n2), np.float32)
    
    # Inverse FFT
    plan_backward = cu_fft.Plan((n1, n2), np.complex64, np.float32)
    cu_fft.ifft(ygpu, x, plan_backward)
    
    # Must divide by the total number of pixels in the image to get the normalisation right
    xout = x.get()/n1/n2
    
    return xout

def runDeconvolutionGPU(sinogram, pixel_size, noise_level=0.05):

    # Subtract the mean from the sinogram
    sinogram -= -np.mean(sinogram)

    # Calculate the FT of the sinogram 
    #fsino = np.fft.fft2(sino)
    fsino = fft2_gpu(sinogram)
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

    #sino_dec = np.real(np.fft.ifft2(fsinot) )
    sino_dec = np.real(ifft2_gpu(fsino_dec) )

    return sino_dec



