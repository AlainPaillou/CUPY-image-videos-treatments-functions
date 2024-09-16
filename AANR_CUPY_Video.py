# coding=utf-8
import time
import numpy as np
import cv2
import cupy as cp


#########################################################################################################
#  Adaptative Absorber Noise Removal Filter (AANRF) - Test program - Copyright Alain Paillou 2018-2024  #
#########################################################################################################

###########################################################################################################
# Licence : this program or parts of program are free to use for personal and non commercial usage only   #
#                                                                                                         #
# If you want to use this program or parts of program for professional or commercial use, you have to ask #
# me before                                                                                               #
#                                                                                                         #
###########################################################################################################

# This program applies an adaptative absorber noise removal filter to a RGB video (0-255 values for the pixels)

# Select the input video for treatment
Video_Test = 'YourVideoName.avi' # The path to your video and video name

# Select the output result video
VideoResult = 'AANR_Result.avi' # The path to your result video

# Choose quality of the output video
flag_HQ = 0 #  0 : low quality compressed video -  1 : high quality RAW video

# Set the dynamic response of the AADF
flag_dyn_AADF = 1 # Choose the filter dynamic - 0 means low dynamic - 1 means high dynamic

flag_ghost_reducer = 0 # only available if Low dynamic filter ; if set to 0 : no ghost treatment, if set to 1 : ghost treatment
val_ghost_reducer = 50 # if ghost reducer is active, choose the threshold to apply ghost reduction ; no treatment applied below threshold value


# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)

# CUDA AANRF function #
adaptative_absorber_noise_removal_Color = cp.RawKernel(r'''
extern "C" __global__
void adaptative_absorber_noise_removal_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int flag_dyn_AADF, int flag_ghost_reducer, int val_ghost_reducer)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;
  int flag_r,flag_g,flag_b;
  float coef_r,coef_g,coef_b;
  
  flag_r = 0;
  flag_g = 0;
  flag_b = 0;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];
      if (flag_dyn_AADF == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (abs(delta_g) > val_ghost_reducer) {
              flag_g = 1;
              dest_g[index] = img_g[index];
          }
          if (abs(delta_b) > val_ghost_reducer) {
              flag_b = 1;
              dest_b[index] = img_b[index];
          }
          if (delta_r > 0 && flag_dyn_AADF == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AADF == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          if (delta_g > 0 && flag_dyn_AADF == 1 && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.025995987)*1.2669433195)));
          }
          if ((delta_g < 0 || flag_dyn_AADF == 0) && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.54405)*20.8425))); 
          }
          if (delta_b > 0 && flag_dyn_AADF == 1 && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.025995987)*1.2669433195)));
          }
          if ((delta_b < 0 || flag_dyn_AADF == 0) && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AADF == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          if (delta_g > 0 && flag_dyn_AADF == 1) {
              coef_g = __powf(abs(delta_g),-0.025995987)*1.2669433195;
          }
          else {
              coef_g = __powf(abs(delta_g),-0.54405)*20.8425; 
          }
          if (delta_b > 0 && flag_dyn_AADF == 1) {
              coef_b = __powf(abs(delta_b),-0.025995987)*1.2669433195;
          }
          else {
              coef_b = __powf(abs(delta_b),-0.54405)*20.8425;
          }
          dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
          dest_g[index] = (int)((old_g[index] - delta_g / coef_g));
          dest_b[index] = (int)((old_b[index] - delta_b / coef_b));
      } 
      }
}
''', 'adaptative_absorber_noise_removal_Color_C')

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
compteur_AANRF = 0
Im1aanrOK = False
Im2aanrOK = False
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

print("AANR Treatment start")
start_time_test = time.perf_counter()
with s1 :
    while Flag_Video_Open == True :
        if (videoIn.isOpened()):
            ret,frame = videoIn.read()
            if ret == True :
                if First_frame == True :
                    First_frame = False
                    res_b1,res_g1,res_r1 = numpy_RGBImage_2_cupy_separateRGB(frame)
                if compteur_AANRF < 3 :
                    compteur_AANRF = compteur_AANRF + 1
                    if compteur_AANRF == 1 :
                        res_b2 = res_b1.copy()
                        res_g2 = res_g1.copy()
                        res_r2 = res_r1.copy()
                        Im1aanrOK = True
                    if compteur_AANRF == 2 :
                        Im2aanrOK = True
                res_b1,res_g1,res_r1 = numpy_RGBImage_2_cupy_separateRGB(frame)
                b_gpu = res_b1
                g_gpu = res_g1
                r_gpu = res_r1
                if Im2aanrOK == True :
                    adaptative_absorber_noise_removal_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                        np.int_(width), np.int_(height),np.intc(flag_dyn_AADF),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))
                    res_b2 = res_b1.copy()
                    res_g2 = res_g1.copy()
                    res_r2 = res_r1.copy()
                    res_r1 = r_gpu.copy()
                    res_g1 = g_gpu.copy()
                    res_b1 = b_gpu.copy()   
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
