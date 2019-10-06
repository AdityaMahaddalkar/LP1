import os
from numba import cuda, vectorize
import cv2
from timeit import default_timer as timer
import numpy as np


@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_y_channel(r_channel, g_channel, b_channel):
    return 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_u_channel(r_channel, g_channel, b_channel):
    return -0.169 * r_channel - 0.331 * g_channel + 0.499 * b_channel + 128

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_v_channel(r_channel, g_channel, b_channel):
    return 0.499 * r_channel - 0.418 * g_channel - 0.0813 * b_channel + 128


def convert_frame_gpu(image):
    
    height, width, depth = image.shape
    
    red_channel = image[:, :, 2]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 0]

    ## Change the shapes of channel vectors
    red_channel = red_channel.reshape(height * width,)
    green_channel = green_channel.reshape(height * width)
    blue_channel = blue_channel.reshape(height * width)
    
    red_channel = red_channel.astype('float32')
    green_channel = red_channel.astype('float32')
    blue_channel = red_channel.astype('float32')

    
    y_channel = ret_y_channel(red_channel, green_channel, blue_channel)
    u_channel = ret_u_channel(red_channel, green_channel, blue_channel)
    v_channel =  ret_v_channel(red_channel, green_channel, blue_channel)

    y_channel = y_channel.reshape(height, width)
    u_channel = u_channel.reshape(height, width)
    v_channel = v_channel.reshape(height, width)

    yuv = np.zeros_like(image)
    yuv[:, :, 0] = y_channel
    yuv[:, :, 1] = u_channel
    yuv[:, :, 2] = v_channel

    return yuv

def convert_images_gpu(source_dir, destination_dir):

    #source_dir = 'resources/'
    #destination_dir = 'outputGPU/'

    list_of_files = list(os.walk(source_dir))[0][2]

    if(len(list_of_files) == 0):
        raise Exception('Files not found. Verify source and destination directory entry')

    ## Timing the CPU (numpy version)
    start = timer()
    for file_name in list_of_files:

        try:
            image = cv2.imread(source_dir + file_name)
            height, width, depth = image.shape
            
            red_channel = image[:, :, 2]
            green_channel = image[:, :, 1]
            blue_channel = image[:, :, 0]

            ## Change the shapes of channel vectors
            red_channel = red_channel.reshape(height * width,)
            green_channel = green_channel.reshape(height * width)
            blue_channel = blue_channel.reshape(height * width)
            
            red_channel = red_channel.astype('float32')
            green_channel = red_channel.astype('float32')
            blue_channel = red_channel.astype('float32')

            y_channel = ret_y_channel(red_channel, green_channel, blue_channel)
            u_channel = ret_u_channel(red_channel, green_channel, blue_channel)
            v_channel =  ret_v_channel(red_channel, green_channel, blue_channel)

            y_channel = y_channel.reshape(height, width)
            u_channel = u_channel.reshape(height, width)
            v_channel = v_channel.reshape(height, width)

            yuv = np.zeros_like(image)
            yuv[:, :, 0] = y_channel
            yuv[:, :, 1] = u_channel
            yuv[:, :, 2] = v_channel

            cv2.imwrite(destination_dir+file_name, yuv)
        except Exception as e:
            print(f'File {file_name} not accessible')
            print(e)

    end = timer()
    print(f'Processed total {len(list_of_files)} files')
    print(f'Total time for GPU {end - start} seconds')

def gpu_video_convert(in_file_name, out_file_name):

    start = timer()
    cap = cv2.VideoCapture(in_file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file_name, fourcc, 20.0, (640, 480))

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == False:
            break

        yuv = convert_frame_gpu(frame)

        out.write(yuv)

        #cv2.imshow('frame', yuv)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    end = timer()
    print(f'Total time for GPU to convert video {end - start} seconds')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
