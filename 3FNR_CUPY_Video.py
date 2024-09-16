# coding=utf-8
import time
import numpy as np
import cv2
import cupy as cp


##############################################################################################
#  3 Frames Noise Removal Filter (AANRF) - Test program - Copyright Alain Paillou 2018-2024  #
##############################################################################################

###########################################################################################################
# Licence : this program or parts of program are free to use for personal and non commercial usage only   #
#                                                                                                         #
# If you want to use this program or parts of program for professional or commercial use, you have to ask #
# me before                                                                                               #
#                                                                                                         #
###########################################################################################################

# This program applies 3 frames noise removal filter to a RGB video (0-255 values for the pixels)



# Select the input video for treatment
Video_Test = 'YourVideoName.avi' # The path to your video and video name

# Select the output result video
VideoResult = '3FNR_Result.avi' # The path to your result video

# Choose quality of the output video
flag_HQ = 0 #  0 : low quality compressed video -  1 : high quality RAW video


# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)

# CUDA 3FNR function #
FNR_Color = cp.RawKernel(r'''
extern "C" __global__
void FNR_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *im1_r, unsigned char *im1_g, unsigned char *im1_b, unsigned char *im2_r, unsigned char *im2_g, unsigned char *im2_b,
unsigned char *im3_r, unsigned char *im3_g, unsigned char *im3_b,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int D1r,D1g,D1b;
  int D2r,D2g,D2b;
  int Delta_r,Delta_g,Delta_b;
  
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    D1r = im2_r[index] - im1_r[index];
    D1g = im2_g[index] - im1_g[index];
    D1b = im2_b[index] - im1_b[index];
    D2r = im3_r[index] - im2_r[index];
    D2g = im3_g[index] - im2_g[index];
    D2b = im3_b[index] - im2_b[index];
  
    if ((D1r*D2r) < 0) {
        Delta_r = (D1r + D2r) / (2.5 - abs(D2r)/255.0);
    }
    else {
        Delta_r = (D1r + D2r) / 2.0;
    }
    if ((D1g*D2g) < 0) {
        Delta_g = (D1g + D2g) / (2.5 - abs(D2g)/255.0);
    }
    else {
        Delta_g = (D1g + D2g) / 2.0;
    }
    if ((D1b*D2b) < 0) {
        Delta_b = (D1b + D2b) / (2.5 - abs(D2b)/255.0);
    }
    else {
        Delta_b = (D1b + D2b) / 2.0;
    }
    if (abs(D2r) > 40) {
        dest_r[index] = im3_r[index];
    }
    else {
        dest_r[index] = (int)(min(max(int((im1_r[index] + im2_r[index]) / 2.0 + Delta_r), 0), 255));
    }
    if (abs(D2g) > 40) {
        dest_g[index] = im3_g[index];
    }
    else {
        dest_g[index] = (int)(min(max(int((im1_g[index] + im2_g[index]) / 2.0 + Delta_g), 0), 255));
    }
    if (abs(D2b) > 40) {
        dest_b[index] = im3_b[index];
    }
    else {
        dest_b[index] = (int)(min(max(int((im1_b[index] + im2_b[index]) / 2.0 + Delta_b), 0), 255));
    }
  }
}
''', 'FNR_Color_C')


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


# Variables setup
First_frame = True
FNR_First_Start = True
compteur_3FNR = 0
img1_3FNROK = False
img2_3FNROK = False
img3_3FNROK = False
Flag_Video_Open = True

# Init Video Input and output
videoIn = cv2.VideoCapture(Video_Test)
property_id = int(cv2.CAP_PROP_FRAME_WIDTH)
width = int(cv2.VideoCapture.get(videoIn, property_id))
property_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
height = int(cv2.VideoCapture.get(videoIn, property_id))

if flag_HQ == 0:
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # compressed video
else :
    fourcc = 0 # RAW video
videoOut = cv2.VideoWriter(VideoResult, fourcc, 25, (width, height), isColor = True) # Compressed video

nb_ThreadsX = 16
nb_ThreadsY = 16
nb_blocksX = (width // nb_ThreadsX) + 1
nb_blocksY = (height // nb_ThreadsY) + 1

print("3FNR Treatment start")
start_time_test = time.perf_counter()

with s1 :
    while Flag_Video_Open == True :
        if (videoIn.isOpened()):
            ret,frame = videoIn.read()
            if ret == True :
                if First_frame == True :
                    First_frame = False
                    res_b1,res_g1,res_r1 = numpy_RGBImage_2_cupy_separateRGB(frame)
                if compteur_3FNR < 4 and FNR_First_Start == True:
                    compteur_3FNR = compteur_3FNR + 1
                    if compteur_3FNR == 1 :
                        imgb1 = res_b1.copy()
                        imgg1 = res_g1.copy()
                        imgr1 = res_r1.copy()
                        img1_3FNROK = True
                    if compteur_3FNR == 2 :
                        imgb2 = res_b1.copy()
                        imgg2 = res_g1.copy()
                        imgr2 = res_r1.copy()
                        img2_3FNROK = True
                    if compteur_3FNR == 3 :
                        imgb3 = res_b1.copy()
                        imgg3 = res_g1.copy()
                        imgr3 = res_r1.copy()
                        img3_3FNROK = True
                        FNR_First_Start = True
                if img3_3FNROK == True :
                    if FNR_First_Start == False :
                        res_b1,res_g1,res_r1 = numpy_RGBImage_2_cupy_separateRGB(frame)
                        imgb3 = res_b1.copy()
                        imgg3 = res_g1.copy()
                        imgr3 = res_r1.copy()   
                    FNR_First_Start = False
                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                
                    FNR_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr1, imgg1, imgb1, imgr2, imgg2, imgb2,\
                                        imgr3, imgg3, imgb3, np.int_(width), np.int_(height)))
                
                    res_r1 = r_gpu.copy()
                    res_g1 = g_gpu.copy()
                    res_b1 = b_gpu.copy()
                    imgb1 = imgb2.copy()
                    imgg1 = imgg2.copy()
                    imgr1 = imgr2.copy()
                    imgb2 = imgb3.copy()
                    imgg2 = imgg3.copy()
                    imgr2 = imgr3.copy()                                    
                    Result_image=cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
                    videoOut.write(Result_image)
            else :
                Flag_Video_Open = False	

videoIn.release()
videoOut.release()

stop_time_test = time.perf_counter()

print("Treatment is OK")

time_exec_test= (stop_time_test-start_time_test)
print("Treatment time : ",time_exec_test," seconds")
