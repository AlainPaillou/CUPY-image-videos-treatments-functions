# Some useful functions for images and videos treatments

*********
Licence :
*********
Fast_NLM2_CUPY_Image_Colour.py and KNN_CUPY_Image_Colour.py code are free of use for any kind of use.

For all other programs and code :
They are free of use for personal use only. Any kind of profesional or/and commercial use of those codes or parts of codes is not allowed. If you want to use those codes or parts of codes for profesional or/and commercial uses, you will have to ask my permission.

**************************
You will have to install :
- CUDA SDK (you will need a NVidia GPU)
- Python (i recommend Python 3.11)
- Numpy
- OpenCV (no need to get a CUDA version of OpenCV)
- Cupy (see : https://cupy.dev/)

All those programs use CUPY and CUPY raw kernels to perform GPU computing in order to speedup treatments time

************
2024-09-16 :
************

__Fast_NLM2_CUPY_Image_Colour.py :__

This programm performs a Fast NLM2 noise removal filter on an image. It could also be use for video treatment

__KNN_CUPY_Image_Colour.py :__

This programm performs a KNN noise removal filter on an image. It could also be use for video treatment

__AANR_CUPY_Video.py :__

This program performs an adaptative absorber noise removal filter on a video. This filter can only be used with a video because it needs 2 consecutives frames to perform the treatment. It is a personal filter which perform really great with quite static video.

__3FNR_CUPY_Video.py :__

This program performs a 3 frames noise removal filter on a video. This filter can only be used with a video because it needs 3 consecutives frames to perform the treatment. It is a personal filter which perform really great with quite static video.

__RAW_Video_debayer_CUPY.py :__

This programs apply a debayer filter on a RAW video (bayer matrix, 1 channel). The debayer result is more precise than classic OpenCV debayer function

__Fast_Image_Colour_Saturation_CUPY.py :__

This program performs a colour saturation enhancement on a colour image. The colour enhancement can be set very high without destroying sharpness and detais of the base image. It is a personal filter which perform really great

