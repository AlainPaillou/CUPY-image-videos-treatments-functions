# coding=utf-8

import time
import cv2
import numpy as np
import cupy as cp
from cupyx.scipy import ndimage


#############################################################################################################
#  Fast image colour saturation enhancement using CUPY  - Test program - Copyright Alain Paillou 2018-2024  #
#############################################################################################################

###########################################################################################################
# Licence : this program or parts of program are free to use for personal and non commercial usage only   #
#                                                                                                         #
# If you want to use this program or parts of program for professional or commercial use, you have to ask #
# me before                                                                                               #
#                                                                                                         #
###########################################################################################################

# this filter raises colour saturation without destroying the image sharpness and details - Fast treatment with CUPY & CUDA

In_File = "Your_Colour_Image.jpg"
Out_File_CUPY = "CUPY_Saturation_Result.jpg"


# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)


Saturation_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
long int width, long int height, float val_sat, int flag_neg_sat)
{

    long int j = threadIdx.x + blockIdx.x * blockDim.x;
    long int i = threadIdx.y + blockIdx.y * blockDim.y;
    long int index;
    double R1,G1,B1;
    double X1;
    double r,g,b;
    double C,X,m;
    double cmax,cmin,diff,h,s,v;
    double radian;
    double cosA,sinA;
    double m1,m2,m3,m4,m5,m6,m7,m8,m9;

    index = i * width + j;
  
    if (i < height && j < width) {
        r = img_r[index] / 255.0;
        g = img_g[index] / 255.0;
        b = img_b[index] / 255.0;
        cmax = max(r, max(g, b));
        cmin = min(r, min(g, b));
        diff = cmax - cmin;
        h = -1.0;
        s = -1.0;
        if (cmax == cmin) 
            h = 0; 
        else if (cmax == r) 
            h = fmod(60 * ((g - b) / diff) + 360, 360); 
        else if (cmax == g) 
            h = fmod(60 * ((b - r) / diff) + 120, 360); 
        else if (cmax == b) 
            h = fmod(60 * ((r - g) / diff) + 240, 360); 
  
        if (cmax == 0) 
            s = 0; 
        else
            s = (diff / cmax); 

        v = cmax;

        s = s * val_sat;

            
        if (h > 360)
            h = 360;
        if (h < 0)
            h = 0;
        if (s > 1.0)
            s = 1.0;
        if (s < 0)
            s = 0;

        C = s*v;
        X = C*(1-abs(fmod(h/60.0, 2)-1));
        m = v-C;

        if(h >= 0 && h < 60){
            r = C,g = X,b = 0;
        }
        else if(h >= 60 && h < 120){
            r = X,g = C,b = 0;
        }
        else if(h >= 120 && h < 180){
            r = 0,g = C,b = X;
        }
        else if(h >= 180 && h < 240){
            r = 0,g = X,b = C;
        }
        else if(h >= 240 && h < 300){
            r = X,g = 0,b = C;
        }
        else{
            r = C,g = 0,b = X;
        }

        R1 = (int)((r+m)*255);
        G1 = (int)((g+m)*255);
        B1 = (int)((b+m)*255);

        if (flag_neg_sat == 1) {
            radian = 3.141592;
            cosA = cos(radian);
            sinA = sin(radian);
            m1 = cosA + (1.0 - cosA) / 3.0;
            m2 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m3 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m4 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m5 = cosA + 1./3.*(1.0 - cosA);
            m6 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m7 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m8 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m9 = cosA + 1./3. * (1.0 - cosA);
            dest_r[index] = (int)(min(max(int(R1 * m1 + G1 * m2 + B1 * m3), 0), 255));
            dest_g[index] = (int)(min(max(int(R1 * m4 + G1 * m5 + B1 * m6), 0), 255));
            dest_b[index] = (int)(min(max(int(R1 * m7 + G1 * m8 + B1 * m9), 0), 255));
        }
        else {
            dest_r[index] = (int)(min(max(int(R1), 0), 255));
            dest_g[index] = (int)(min(max(int(G1), 0), 255));
            dest_b[index] = (int)(min(max(int(B1), 0), 255));
        }
    }
}
''', 'Saturation_Colour_C')


Saturation_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *ext_r, unsigned char *ext_g, unsigned char *ext_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]) / (0.299*ext_r[index] + 0.587*ext_g[index] + 0.114*ext_b[index]);
      dest_r[index] = (int)(min(max(int(ext_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(ext_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(ext_b[index]*X), 0), 255));
    } 
}
''', 'Saturation_Combine_Colour_C')


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

# Realize a Gaussian blur with CUPY function
def gaussianblur_colour(im_r,im_g,im_b,niveau_blur):
    im_GB_r = ndimage.gaussian_filter(im_r, sigma = niveau_blur)
    im_GB_g = ndimage.gaussian_filter(im_g, sigma = niveau_blur)
    im_GB_b = ndimage.gaussian_filter(im_b, sigma = niveau_blur)
    return im_GB_r,im_GB_g,im_GB_b


image_brut_CV = cv2.imread(In_File,cv2.IMREAD_COLOR)
height,width,layers = image_brut_CV.shape
nb_pixels = height * width


# Set saturation parameter
val_SAT = 10 # from 0 which gives B&W result to 30 ; 1 means no saturation applied

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
    r_gpu = res_r.copy()
    g_gpu = res_g.copy()
    b_gpu = res_b.copy()
    init_r = res_r.copy()
    init_g = res_g.copy()
    init_b = res_b.copy()
    coul_r,coul_g,coul_b = gaussianblur_colour(r_gpu,g_gpu,b_gpu,3)
    flag_neg_sat = 0  
    Saturation_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, coul_r, coul_g, coul_b, np.int_(width), np.int_(height),np.float32(val_SAT), np.int_(flag_neg_sat)))
    coul_gauss2_r = r_gpu.copy()
    coul_gauss2_g = g_gpu.copy()
    coul_gauss2_b = b_gpu.copy()               
    coul_gauss2_r,coul_gauss2_g,coul_gauss2_b = gaussianblur_colour(coul_gauss2_r,coul_gauss2_g,coul_gauss2_b,7)
    Saturation_Combine_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, init_r, init_g, init_b, coul_gauss2_r,coul_gauss2_g,coul_gauss2_b, np.int_(width), np.int_(height)))
    res_r = r_gpu.copy()
    res_g = g_gpu.copy()
    res_b = b_gpu.copy()
    image_result_GPU=cupy_separateRGB_2_numpy_RGBimage(res_r,res_g,res_b)

tps_GPU = time.time() - tps1

print("CUPY treatment OK")
print ("GPU treatment time : ",tps_GPU)
print("")

cv2.imwrite(Out_File_CUPY, cv2.cvtColor(image_result_GPU, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format

