# coding=utf-8

import time
import cv2
import numpy as np
import cupy as cp


##################################################################################
#  KNN Noise Removal Filter  - Test program - Copyright Alain Paillou 2018-2024  #
##################################################################################

##################################################################################
# Licence : this program or parts of program are free to use for any kind of use #
#                                                                                #
# KNN Denoise filter for colour image V1.1                                       #
# Based on KNN filter CUDA program provided by Nvidia - CUDA SDK samples         #
#                                                                                #
##################################################################################


In_File = "Your_Colour_Image.jpg"
Out_File_CUPY = "CUPY_KNN_Result.jpg"


# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)


KNN_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void KNN_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        float3 clr00 = {0, 0, 0};
        float3 clrIJ = {0, 0, 0};
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4];
        clr00.y = img_g[index4];
        clr00.z = img_b[index4];
    
        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index2];
                clrIJ.y = img_g[index2];
                clrIJ.z = img_b[index2];
                float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'KNN_Colour_C')


# Convert a numpy RGB image to 3 CUPY arrays R, G and B
def numpy_RGBImage_2_cupy_separateRGB(numpyImageRGB):
    cupyImageRGB = cp.asarray(numpyImageRGB)
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    return cupy_R,cupy_G,cupy_B

# Convert 3 CUPY array R, G & B toa numpy RGB image
def cupy_separateRGB_2_numpy_RGBimage(cupyR,cupyG,cupyB):
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    numpyRGB = cupyRGB.get()
    return numpyRGB


image_brut_CV = cv2.imread(In_File,cv2.IMREAD_COLOR)
height,width,layers = image_brut_CV.shape

# Set KNN parameters
KNN_Noise = 0.32
Noise = 1.0/(KNN_Noise*KNN_Noise)
lerpC = 0.2

# Algorithm GPU using CUDA & CUPY

# Set blocks et Grid sizes
nb_ThreadsX = 16
nb_ThreadsY = 16
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

print("Test GPU ",nb_blocksX*nb_blocksY," Blocks ",nb_ThreadsX*nb_ThreadsY," Threads/Block")
tps1 = time.time()

with s1 :
    res_b,res_g,res_r = numpy_RGBImage_2_cupy_separateRGB(image_brut_CV)
    b_gpu = res_b
    g_gpu = res_g
    r_gpu = res_r
    KNN_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r, res_g, res_b, np.intc(width), np.intc(height), np.float32(Noise), \
        np.float32(lerpC)))
    res_r = r_gpu
    res_g = g_gpu
    res_b = b_gpu
    image_result_GPU=cupy_separateRGB_2_numpy_RGBimage(res_r,res_g,res_b)

tps_GPU = time.time() - tps1

print("CUPY treatment OK")
print ("GPU treatment time : ",tps_GPU)
print("")

cv2.imwrite(Out_File_CUPY, cv2.cvtColor(image_result_GPU, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format


