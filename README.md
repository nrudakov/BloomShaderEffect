# BloomShaderEffect
The bloom filter operates as follows: firstly for each pixel calculate its luminance based on RGB intensities; second, extract the highest luminance values from the resulting luminance surface based on a given threshold; finally, apply Gaussian blur onto thresholded luminance surface and add the resulting image with the original one. See how brightly the sun starts to shine!
