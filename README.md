# BloomShaderEffect
The bloom filter operates as follows: 
firstly for each pixel calculate its luminance based on RGB intensities; 
second, extract the highest luminance values from the resulting luminance surface based on a given threshold; 
finally, apply Gaussian blur onto thresholded luminance surface and add the resulting image with the original one. 
See how brightly the sun starts to shine!

To implement this method the CUDA Toolkit of version 9.1 was used together with Python 3.6 interpreter and PyCUDA 2017.1.1 library.
This implementation is a single Python script that contains both CPU and GPU variants of the bloom algorithm. The script accepts one obligatory argument — the filename of the image to bloom, and six optional arguments: luminance threshold, Gaussian blur standard deviation, number of standard deviation taken for blurring, desired mode (GPU, CPU or both) to run, filename to save resulting image, and an option to show or not the results in a figure.

To view help run: python bloom.py --help
