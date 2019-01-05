import astra
import numpy as np

def iradon_astra(sinogram, theta, output_size):

    ''' Implementation of the FBP algorithm using GPU accelerated FBP_CUDA
    from the astra toolbox.
    The output is fully compatible with skimage.transform.iradon'''
    
    # The proj_geom of astra needs a sino in the shape (angles, pixels)
    sino = np.transpose(sinogram)
    
    # Convert angles in rad
    theta = 2*np.pi*theta/360.

    # Create geometries and initialise projectors
    vol_geom = astra.create_vol_geom(output_size, output_size)
    proj_geom = astra.create_proj_geom('parallel', 1.0, sino.shape[1], theta)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    sinogram_id = astra.data2d.create('-sino', proj_geom, sino)

    # Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    # Create configuration 
    cfg = astra.astra_dict('FBP_CUDA')
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectionDataId'] = sinogram_id


    # Create and run the algorithm object from the configuration structure
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    # Get the result
    rec = astra.data2d.get(rec_id)

    # Clean up. Note that GPU memory is tied up in the algorithm object,
    # and main RAM in the data objects.
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(sinogram_id)
    astra.projector.delete(proj_id)

    return rec
