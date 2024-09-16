# coding=utf-8
import time
import numpy as np
import cv2
import cupy as cp


############################################################################################
#  RAW video debayer using CUPY & CUDA - Test program - Copyright Alain Paillou 2018-2024  #
############################################################################################

###########################################################################################################
# Licence : this program or parts of program are free to use for personal and non commercial usage only   #
#                                                                                                         #
# If you want to use this program or parts of program for professional or commercial use, you have to ask #
# me before                                                                                               #
#                                                                                                         #
###########################################################################################################

# RAW video debayer to produce RGB video (0-255 values for the pixels) - the reasult is more accurate than classical OpenCV debayer

# Select the input video for treatment
Video_Test = 'YourRAWVideoName.avi' # The path to your video and video name

# Select the output result video
VideoResult = 'DEBAYER_Result.avi' # The path to your result video

# Choose quality of the output video
flag_HQ = 0 #  0 : low quality compressed video -  1 : high quality RAW video


# Saving CUPY context
s1 = cp.cuda.Stream(non_blocking=False)

# CUDA DEBAYER function #
Image_Debayer_GPU = cp.RawKernel(r'''
extern "C" __global__
void Image_Debayer_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img, long int width, long int height, int GPU_BAYER)
{

  long int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  long int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  float att;
  
  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;
  i5 = (i-1) * width + j-1;
  i6 = (i-1) * width + j;
  i7 = (i-1) * width + j+1;
  i8 = (i-1) * width + j+2;
  i9 = i * width + j+2;
  i10 = (i+1) * width + j+2;
  i11 = (i+2) * width + j+2;
  i12 = (i+2) * width + j+1;
  i13 = (i+2) * width + j;
  i14 = (i+2) * width + j-1;
  i15 = (i+1) * width + j-1;
  i16 = i * width + j-1;
  att = 1 / 4.0;
  
  if (i < (height-1) && i > 0 && j < (width-1) && j > 0) {
// RGGB
      if (GPU_BAYER == 1) {
          dest_r[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_b[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_r[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_b[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_r[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_b[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_r[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_b[i4]=img[i4];
          }
// BGGR
      if (GPU_BAYER == 2) {
          dest_b[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_r[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_b[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_r[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_b[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_r[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_b[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_r[i4]=img[i4];
          }
// GBRG
      if (GPU_BAYER == 3) {
          dest_r[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_b[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_r[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_b[i2]=img[i2];

          dest_r[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_b[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_r[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_b[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
// GRBG
      if (GPU_BAYER == 4) {
          dest_b[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_r[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_b[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_r[i2]=img[i2];

          dest_b[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_r[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_b[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_r[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
    }
}
''', 'Image_Debayer_GPU_C')


# Convert 3 CUPY array R, G & B toa numpy RGB image
def cupy_separateRGB_2_numpy_RGBimage(cupyR,cupyG,cupyB):
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    numpyRGB = cupyRGB.get()
    return numpyRGB

# Variables setup
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
nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
nb_blocksY = ((height // 2) // nb_ThreadsY) + 1

print("DEBAYER Treatment start")
start_time_test = time.perf_counter()

# Choose BAYER pattern
#GPU_BAYER = 1 # RGGB
#GPU_BAYER = 2 # BGGR
#GPU_BAYER = 3 # GBRG
GPU_BAYER = 4 # GRBG

with s1 :
    while Flag_Video_Open == True :
        if (videoIn.isOpened()):
            ret,frame = videoIn.read()
            if ret == True :
                mono = frame[:,:,0].copy() # frame is read as a color array so we need to take only one channel
                img = cp.asarray(mono,dtype=cp.uint8)
                r_gpu = cp.zeros_like(mono,dtype=cp.uint8)
                g_gpu = cp.zeros_like(mono,dtype=cp.uint8)
                b_gpu = cp.zeros_like(mono,dtype=cp.uint8)
                Image_Debayer_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, img, np.intc(width), np.intc(height), np.intc(GPU_BAYER)))
                res_r = r_gpu.copy()
                res_g = g_gpu.copy()
                res_b = b_gpu.copy()   
                Result_image=cupy_separateRGB_2_numpy_RGBimage(res_b,res_g,res_r)
                videoOut.write(Result_image)
            else :
                Flag_Video_Open = False	

videoIn.release()
videoOut.release()

stop_time_test = time.perf_counter()

print("Treatment is OK")

time_exec_test= (stop_time_test-start_time_test)
print("Treatment time : ",time_exec_test," seconds")
