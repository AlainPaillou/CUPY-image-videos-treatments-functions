# coding=utf-8

import time
import cv2
import numpy as np
import cupy as cp


########################################################################################
#  Fast NLM2 Noise Removal Filter  - Test program - Copyright Alain Paillou 2018-2024  #
########################################################################################

##################################################################################
# Licence : this program or parts of program are free to use for any kind of use #
#                                                                                #
# Fast NLM Denoise filter for colour image V1.1                                  #
# Based on Fast NLM filter CUDA program provided by Nvidia - CUDA SDK samples    #
#                                                                                #
##################################################################################


In_File = "Your_Colour_Image.jpg"
Out_File_OpenCV = "OpenCV_NLM2_Result.jpg"
Out_File_CUPY = "CUPY_NLM2_Result.jpg"

# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)


# CUDA Fast NLM2 noise removal function
NLM2_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void NLM2_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = 6;
    const float limxmax = imageW - 6;
    const float limymin = 6;
    const float limymax = imageH - 6;
   
    long int index4;
    long int index5;

    if(x>limxmin && x<limxmax && y>limymin && y<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])
                + (img_g[index2] - img_g[index1]) * (img_g[index2] - img_g[index1])
                + (img_b[index2] - img_b[index1]) * (img_b[index2] - img_b[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0.0, 0.0, 0.0};

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float3 clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index3];
                clrIJ.y = img_g[index3];
                clrIJ.z = img_b[index3];
                
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float3 clr00 = {0.0, 0.0, 0.0};
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4] / 256.0;
        clr00.y = img_g[index4] / 256.0;
        clr00.z = img_b[index4] / 256.0;
        
        clr.x = clr.x + (clr00.x - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'NLM2_Colour_C')

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
nb_pixels = height * width


# Set NLM2 parameters
NLM_Noise = 1.45
Noise = 1.0/(NLM_Noise*NLM_Noise)
lerpC = 0.4

# Algorithm CPU using OpenCV fastNlMeansDenoisingColored routine
tps1 = time.time()
param=21.0
image_brut_CPU=cv2.fastNlMeansDenoisingColored(image_brut_CV, None, param, param, 3, 5) # application filtre denoise software colour
image_brut_CPU=cv2.cvtColor(image_brut_CPU, cv2.COLOR_BGR2RGB)
tps_CPU = time.time() - tps1              
print("OpenCV treatment OK")
print ("CPU treatment time : ",tps_CPU)
print("")
cv2.imwrite(Out_File_OpenCV, cv2.cvtColor(image_brut_CPU, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format


# Algorithm GPU using CUDA & CUPY

#  blocks & Grid sizes
nb_ThreadsX = 8
nb_ThreadsY = 8
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

print("Test GPU ",nb_blocksX*nb_blocksY," Blocks ",nb_ThreadsX*nb_ThreadsY," Threads/Block")

# Set NLM2 parameters
NLM_Noise = 1.45
Noise = 1.0/(NLM_Noise*NLM_Noise)
lerpC = 0.4

tps1 = time.time()

with s1 :
    res_b,res_g,res_r = numpy_RGBImage_2_cupy_separateRGB(image_brut_CV)
    b_gpu = res_b
    g_gpu = res_g
    r_gpu = res_r
    NLM2_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r, res_g, res_b, np.intc(width), np.intc(height), np.float32(Noise), \
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


Ecart1 = tps_CPU/tps_GPU
print ("Acceleration factor GPU/CPU : ",Ecart1)
