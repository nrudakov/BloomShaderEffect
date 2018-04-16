import argparse
import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import string
import time
from pycuda.compiler import SourceModule
from scipy import misc
from scipy import ndimage
from scipy.ndimage.filters import gaussian_filter
from textwrap import dedent


# Round a / b to nearest higher integer value.
def int_div_up(a, b):
    dm = divmod(a, b)
    return int(dm[0] + 1) if dm[1] != 0 else int(   dm[0])


# Align 'a' to nearest higher multiple of 'b'.
def int_align_up(a, b):
    return int(a - a % b + b) if a % b != 0 else int(a)


# Apply bloom shader effect to image on GPU using CUDA.
def bloom_gpu(img, threshold, sigma, sigma_n):
    kernel_radius = np.int32(sigma_n * sigma + 0.5)
    poly = np.polynomial.Polynomial([0, 0, -0.5 / (sigma * sigma)])
    x = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(poly(x), dtype=np.float32)
    kernel /= kernel.sum()
    bloom_cuda_source_template = """
        /* 
         * Function:    luminance
         * ----------------------   
         * Performs perceptual luminance-preserving conversion of sRGB image to grayscale image.
         * @param lum: resulting grayscale (1-channel) image as 1D float array;
         *             1D-mapping: from left to right, from top to bottom.
         * @param img: sRGB (3-channel) image as 1D float array;
         *             1D-mapping: from R to B, from left to right, from top to bottom.
         * @param n: total number of pixels in image (length of lum array) as single-element int array.
         */ 
        __global__ void luminance(float *lum, float *img, int *n)
        {
            // Luminance perception coefficients for sRGB.
            const float c_r = 0.2126;
            const float c_g = 0.7152;
            const float c_b = 0.0722;
            
            // Get 1D-mapped index of pixel.
            int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Perform perceptual luminance-preserving conversion of sRGB image to grayscale image
            // (one grayscale pixel per thread)
            // after checking that 1D-mapped index of pixel is less than total number of pixels.
            if (g_idx < n[0])
                lum[g_idx] = c_r * img[g_idx * 3 + 0] + c_g * img[g_idx * 3 + 1] + c_b * img[g_idx * 3 + 2];
        }

        /*
         * Function:    array_max
         * ----------------------
         * Finds the maximum value in 1D array using reduction technique.
         * @param max_val: resulting maximum value in array as single-element float array.
         * @param array: data as 1D float array.
         * @param n: number of elements in array as single-element int array.
         * @param mutex: mutex value (need to be zero) as single-element int array.
         */
        __global__ void array_max(float *max_val, float *array, int *n, int *mutex)
        {
            extern __shared__ float sdata[];
            
            // Load array into shared memory.
            // The number of blocks is halved, so it is possible to load the maximum of two values:
            // the element within current block and the element with index shifted by size of the block.
            const int g_idx = __mul24(blockIdx.x, blockDim.x << 1) + threadIdx.x; 
            float x = -1.0;
            if(g_idx + blockDim.x < n[0])
                x = fmaxf(array[g_idx], array[g_idx + blockDim.x]);
            sdata[threadIdx.x] = x;
            __syncthreads();
            
            // Perform reduction within one block.
            if(threadIdx.x < 512 && sdata[threadIdx.x + 512] > sdata[threadIdx.x])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + 512];
            }
            __syncthreads();
            if(threadIdx.x < 256 && sdata[threadIdx.x + 256] > sdata[threadIdx.x])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + 256];
            }
            __syncthreads();
            if(threadIdx.x < 128 && sdata[threadIdx.x + 128] > sdata[threadIdx.x])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + 128];
            }
            __syncthreads();
            if(threadIdx.x < 64 && sdata[threadIdx.x + 64] > sdata[threadIdx.x])
            {
                sdata[threadIdx.x] = sdata[threadIdx.x + 64];
            }
            __syncthreads();
            
            // Single-warp threads in use now, no thread synchronisation needed. 
            if (threadIdx.x < 32)
            {
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 16]);
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 8]);
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 4]);
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 2]);
                sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + 1]);
            }
            
            // The first thread updates maximum value using mutex to ensure update is correct.
            if(threadIdx.x == 0)
            {
                // Lock mutex
                while(atomicCAS(mutex,0,1) != 0);
                // Write result
                max_val[0] = fmaxf(max_val[0], sdata[0]);
                // Unlock mutex
                atomicExch(mutex, 0);
            }
        }

        /*
         * Function:    array_highpass
         * ---------------------------
         * Zeros elements of array which are less or equal to threshold.
         * @param array: data as 1D float array.
         * @param n: number of elements in array as single-element int array.
         * @param threshold: the maxmum value to zero as single-element float array.
         */
        __global__ void array_highpass(float *array, int *n, float *threshold)
        {
            // Get 1D-mapped index of pixel.
            int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Perform threshold-based zeroing
            // after checking that 1D-mapped index of pixel is less than total number of pixels.
            if (g_idx < n[0])
                if (array[g_idx] <= threshold[0])
                    array[g_idx] = 0.0;
        }

        
        #define kernel_radius           $kernel_radius
        #define kernel_radius_aligned   $kernel_radius_aligned
        #define kernel_w                $kernel_w
        #define row_tile_w              $row_tile_w
        #define col_tile_w              $col_tile_w
        #define col_tile_h              $col_tile_h

        __device__ __constant__ float kernel[kernel_w];

        /*
         * Function:    convolution_row
         * ---------------------------
         * Performs 1D convolution for image in horizontal direction using predefined above kernel.
         * @param input: data as 1D float array.
         * @param dataW: width of image in pixels as pointer to int.
         * @param dataH: height of image in pixels as pointer to int.
         */
        __global__ void convolution_row(float *input,
                                        int *dataW,
                                        int *dataH)
        {
            extern __shared__ float data[];
            // Define working area
            const int tile_start = __mul24(blockIdx.x, row_tile_w);
            const int tile_end = tile_start + row_tile_w;
            const int apron_start = tile_start - kernel_radius;
            const int apron_end = tile_end + kernel_radius;
            
            // tile_start is clamped by definition!
            const int tile_end_clamped = min(tile_end, dataW[0] - 1);
            const int apron_start_clamped = max(apron_start, 0);
            const int apron_end_clamped = min(apron_end, dataW[0] - 1);

            const int row_start = __mul24(blockIdx.y, dataW[0]);
            const int apron_start_aligned = tile_start - kernel_radius_aligned;
            const int load_pos = apron_start_aligned + threadIdx.x;
            
            // Transfer data to shared memory
            if (load_pos >= apron_start)
            {
                data[load_pos - apron_start] = 
                    (apron_start_clamped <= load_pos && load_pos <= apron_end_clamped) ?
                        input[row_start + load_pos] : 0;
            }

            __syncthreads();
            // Perform convolution and write result to global memory
            const int write_pos = tile_start + threadIdx.x;
            if (write_pos <= tile_end_clamped)
            {
                const int smem_pos = write_pos - apron_start;
                float sum = 0;
                """
    for k in range(-kernel_radius, kernel_radius + 1):
        bloom_cuda_source_template += string.Template(
            'sum += data[smem_pos + $k] * kernel[kernel_radius - $k];\n').substitute(k=k)
    bloom_cuda_source_template += """
                input[row_start + write_pos] = sum;
            }    
        }

        /*
         * Function:    convolution_column
         * ---------------------------
         * Performs 1D convolution for image in vertical direction using predefined above kernel.
         * @param input: data as 1D float array.
         * @param dataW: width of image in pixels as pointer to int.
         * @param dataH: height of image in pixels as pointer to int.
         * @param smem_stride: stride in shared memory array as pointer to int.
         * @param gmem_stride: stride in global memory array as pointer to int.
         */
        __global__ void convolution_column(float *input,
                                           int *dataW,
                                           int *dataH,
                                           int *smem_stride,
                                           int *gmem_stride)
        {
            extern __shared__ float data[];
            // Define working area
            const int tile_start = __mul24(blockIdx.y, col_tile_h);
            const int tile_end = tile_start + col_tile_h - 1;
            const int apron_start = tile_start - kernel_radius;
            const int apron_end = tile_end + kernel_radius;

            const int tile_end_clamped = min(tile_end, dataH[0] - 1);
            const int apron_start_clamped = max(apron_start, 0);
            const int apron_end_clamped = min(apron_end, dataH[0] - 1);

            const int column_start = __mul24(blockIdx.x, col_tile_w) + threadIdx.x;

            int smem_pos = __mul24(threadIdx.y, col_tile_w) + threadIdx.x;
            int gmem_pos = __mul24(apron_start + threadIdx.y, dataW[0]) + column_start;
            
            // Transfer data from global to shared memory.
            for (int y = apron_start + threadIdx.y; y < apron_end; 
                y += blockDim.y, smem_pos += smem_stride[0], gmem_pos += gmem_stride[0])
            {
                data[smem_pos] = 
                    (apron_start_clamped <= y && y <= apron_end_clamped) ? 
                        input[gmem_pos] : 0;
            }

            __syncthreads();
            // Perform convolution (each thread performs convolution several times to different parts of image)
            smem_pos = __mul24(threadIdx.y + kernel_radius, col_tile_w) + threadIdx.x;
            gmem_pos = __mul24(tile_start + threadIdx.y, dataW[0]) + column_start;
            for (int y = tile_start + threadIdx.y; y <= tile_end_clamped;
                y += blockDim.y, smem_pos += smem_stride[0], gmem_pos += gmem_stride[0])
            {
                float sum = 0;
                """
    for k in range(-kernel_radius, kernel_radius + 1):
        bloom_cuda_source_template += string.Template(
            'sum += data[smem_pos + __mul24($k, col_tile_w)] * kernel[kernel_radius - $k];\n').substitute(k=k)
    bloom_cuda_source_template += """
                input[gmem_pos] = sum;
            }
        }
        /*
         * Function:    array_add_sat255
         * -----------------------------
         * Performs addition with saturation to 255.0 of sRGB (3-channel) image with grayscale (1-channel) image.
         * @param array3: initial and resulting sRGB (3-channel) image as 1D float array;
         *                1D-mapping: from R to B, from left to right, from top to bottom.
         * @param array1: grayscale (1-channel) image as 1D float array;
         *                1D-mapping: from left to right, from top to bottom.
         * @param w: width of image in pixels as single-element int array.
         * @param n: total number of pixels in image (length of array1) as single-element int array.
         */
        __global__ void arrays_add_sat255(float *array3, float *array1, int *w, int *h)
        {
            int px_x = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
            int px_y = __mul24(blockIdx.y, blockDim.y) + threadIdx.y;
            int px = px_y * w[0] + px_x;
            array3[px * 3 + 0] = array3[px * 3 + 0] + array1[px];
            array3[px * 3 + 1] = array3[px * 3 + 1] + array1[px];
            array3[px * 3 + 2] = array3[px * 3 + 2] + array1[px];
            if (array3[px * 3 + 0] > 255.0)
                array3[px * 3 + 0] = 255.0;
            if (array3[px * 3 + 1] > 255.0)
                array3[px * 3 + 1] = 255.0;
            if (array3[px * 3 + 2] > 255.0)
                array3[px * 3 + 2] = 255.0;
        }
    """
    kernel_radius_aligned = int(16)
    row_tile_w = int(128)
    col_tile_w = int(16)
    col_tile_h = int(48)
    bloom_cuda_source = string.Template(bloom_cuda_source_template). \
        substitute(kernel_radius=kernel_radius,
                   kernel_radius_aligned=kernel_radius_aligned,
                   kernel_w=kernel.size,
                   row_tile_w=row_tile_w,
                   col_tile_w=col_tile_w,
                   col_tile_h=col_tile_h)
    bloom_cuda_module = SourceModule(bloom_cuda_source)

    img = img.astype(np.float32)
    sat = np.empty_like(img)
    w = np.int32(img.shape[1])
    h = np.int32(img.shape[0])
    n = np.int32(w * h);
    w_al = np.int32(int_align_up(w, 16))
    smem_stride = np.int32(col_tile_w * 8)
    gmem_stride = np.int32(w_al * 8)
    lum = np.zeros((h, w), np.float32)
    img_g = drv.mem_alloc(img.nbytes)
    lum_g = drv.mem_alloc(lum.nbytes)
    kernel_g = bloom_cuda_module.get_global('kernel')[0]
    w_g = drv.mem_alloc(4)
    h_g = drv.mem_alloc(4)
    n_g = drv.mem_alloc(4)
    w_al_g = drv.mem_alloc(4)
    smem_stride_g = drv.mem_alloc(4)
    gmem_stride_g = drv.mem_alloc(4)
    drv.memcpy_htod(img_g, img)
    drv.memcpy_htod(lum_g, lum)
    drv.memcpy_htod(kernel_g, kernel)
    drv.memcpy_htod(w_g, w)
    drv.memcpy_htod(h_g, h)
    drv.memcpy_htod(n_g, n)
    drv.memcpy_htod(w_al_g, w_al)
    drv.memcpy_htod(smem_stride_g, smem_stride)
    drv.memcpy_htod(gmem_stride_g, gmem_stride)

    block_size = 1024
    grid_size = int_div_up(n, block_size)
    bloom_cuda_module.get_function("luminance")(
        lum_g, img_g, n_g,
        block=(block_size, 1, 1),
        grid=(grid_size, 1, 1))
    lum_max = np.zeros((1, 1), np.float32)
    mutex = np.zeros((1, 1), np.int32)
    bloom_cuda_module.get_function("array_max")(
        drv.Out(lum_max), lum_g, n_g, drv.In(mutex),
        block=(block_size, 1, 1),
        grid=(grid_size >> 1, 1),
        shared=block_size * 4)
    lum_threshold = np.float32(lum_max * threshold)
    bloom_cuda_module.get_function("array_highpass")(
        lum_g, n_g, drv.In(lum_threshold),
        block=(block_size, 1, 1),
        grid=(grid_size, 1))
    block_size_x = int(kernel_radius_aligned + row_tile_w + kernel_radius)
    grid_size_x = int(int_div_up(w_al, row_tile_w))
    grid_size_y = int(h)
    shared_size = int((kernel_radius + row_tile_w + kernel_radius) * 4)
    bloom_cuda_module.get_function("convolution_row")(
        lum_g, w_al_g, h_g,
        block=(block_size_x, 1, 1),
        grid=(grid_size_x, grid_size_y),
        shared=shared_size)
    grid_size_x = int(int_div_up(w_al, col_tile_w))
    grid_size_y = int(int_div_up(h, col_tile_h))
    shared_size = int(col_tile_w * (kernel_radius + row_tile_w + kernel_radius) * 4)
    bloom_cuda_module.get_function("convolution_column")(
        lum_g, w_al_g, h_g, smem_stride_g, gmem_stride_g,
        block=(col_tile_w, 8, 1),
        grid=(grid_size_x, grid_size_y),
        shared=shared_size)
    bloom_cuda_module.get_function("arrays_add_sat255")(
        img_g, lum_g, w_g, h_g,
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    drv.memcpy_dtoh(sat, img_g)
    sat = sat.astype(np.uint8)
    return sat


# Apply bloom shader effect to image on CPU.
def bloom_cpu(img, threshold, sigma, sigma_n):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    blurred = gaussian_filter(lum * (lum > threshold * ndimage.maximum(lum)), sigma, truncate=sigma_n)
    r_res, g_res, b_res = r + blurred, g + blurred, b + blurred
    np.clip(r_res, 0, 255, out=r_res)
    np.clip(g_res, 0, 255, out=g_res)
    np.clip(b_res, 0, 255, out=b_res)
    return np.dstack((np.uint8(r_res), np.uint8(g_res), np.uint8(b_res)))


# Convert string argument to boolean value
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Parse script arguments
def parse_arguments():
    # Parser initialization
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=dedent(
            'This program applies bloom effect to an image.\n'
            'https://en.wikipedia.org/wiki/Bloom_(shader_effect)\n'
            'There are two implementations:\n'
            '\t- CPU;\n'
            '\t- GPU CUDA.'),
        epilog=dedent(
            'Developed by: Nikolay Rudakov\n'
            '\tas the assignment for BM40A1400 GPGPU Computing course, LUT, 2017.'))
    parser.add_argument('input', type=str,
                        help='Input image: filename of an image to bloom.')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, dest='lum_threshold', metavar='',
                        help='Luminance threshold: the percentage of maximum luminance. '
                             'Pixels above the luminance threshold will be augmented. '
                             'Default value is 0.5.')
    parser.add_argument('-s', '--sigma', type=float, default=8.0, dest='sigma', metavar='',
                        help='Sigma: standard deviation for Gaussian kernel. '
                             'Default value is 8.0.')
    parser.add_argument('-n', '--sigma_n', type=float, default=3.0, dest='sigma_n', metavar='',
                        help='Sigma_n: truncate the filter at this many standard deviations.'
                             'Default value is 3.0.')
    parser.add_argument('-m', '--mode', type=str, choices=['cpu', 'gpu', 'both'], default='both',
                        dest='mode', metavar='',
                        help='Mode: which implementation to use: "cpu" (python numpy), "gpu" (CUDA) or "both". '
                             'In "both" mode only gpu-bloomed image will be saved to file if -o argument is specified. '
                             'Default value is "both".')
    parser.add_argument('-o', '--output', type=str, dest='output', metavar='',
                        help='Output image: filename to save bloomed image. '
                             'In "both" mode gpu-bloomed image only will be saved.')
    parser.add_argument('-u', '--ui', type=str2bool, default=True, dest='ui', metavar='',
                        help='UI: Show initial and result images and execution times.')
    # Execute parsing
    return parser.parse_args()


def main():
    # Parse command line
    args = parse_arguments()
    # Read input image
    img = misc.imread(args.input)
    # Initialize the figure to show images
    figure = plt.figure()
    figure.suptitle('Bloom shader effect: luminance threshold =  ' + str(args.lum_threshold) +
                    ', Gaussian kernel standard deviation = ' + str(args.sigma) +
                    ', standard deviations taken = ' + str(args.sigma_n))
    subplots_total = 3 if args.mode == 'both' else 2
    subplot_current = 1
    # Set original image to subplot
    subplot = figure.add_subplot(1, subplots_total, subplot_current)
    subplot.set_title('Original image')
    subplot_current += 1
    plt.imshow(img, cmap='gray')
    # Print parameters
    print('Bloom shader effect: ' + args.input + ' (' + str(img.shape[1]) + 'x' + str(img.shape[0]) +
          '), t = ' + str(args.lum_threshold) + ', s = ' + str(args.sigma) + ', n = ' + str(args.sigma_n) +
          (', o = ' + args.output if args.output is not None else '') + '\n')
    # Run CPU implementation
    if args.mode == 'cpu' or args.mode == 'both':
        t0 = time.perf_counter()
        img_bloomed_c = bloom_cpu(img, args.lum_threshold, args.sigma, args.sigma_n)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        print('\tCPU elapsed time: ' + str(elapsed_time) + ' seconds')
        # Set CPU-bloomed image to subplot
        subplot = figure.add_subplot(1, subplots_total, subplot_current)
        plt.imshow(img_bloomed_c, cmap='gray')
        subplot.set_title('CPU-bloomed image: %.3f seconds' % elapsed_time)
        subplot_current += 1
    # Run GPU implementation
    if args.mode == 'both' or args.mode == 'gpu':
        t0 = time.perf_counter()
        img_bloomed_g = bloom_gpu(img, args.lum_threshold, args.sigma, args.sigma_n)
        t1 = time.perf_counter()
        elapsed_time = t1 - t0
        print('\tGPU elapsed time: ' + str(elapsed_time) + ' seconds')
        drv.stop_profiler()
        # Set GPU-bloomed image to subplot
        subplot = figure.add_subplot(1, subplots_total, subplot_current)
        plt.imshow(img_bloomed_g, cmap='gray')
        subplot.set_title('GPU-bloomed image: %.3f seconds' % elapsed_time)
        subplot_current += 1
    # Save bloomed image to file
    if args.output is not None:
        if args.mode == 'cpu':
            misc.imsave(args.output, img_bloomed_c)
        else:
            misc.imsave(args.output, img_bloomed_g)
    # Figure original and bloomed images
    if args.ui is True:
        plt.show()


if __name__ == "__main__":
    main()
