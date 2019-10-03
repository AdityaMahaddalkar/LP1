import cv2
import numpy as np
import time
import os
from timeit import default_timer as timer
from numba import cuda, vectorize

conversion_matrix = [[0.299, -0.1687, 0.5000],
                    [0.58700, -0.33126, 0.41869],
                    [0.11400, 0.5000, -0.08131]]
    
conversion_matrix = np.array(conversion_matrix)


def cpu_rbg2yuv(image):
    
    yuv = np.zeros(shape=image.shape)
    
    r_channel = image[:, :, 2]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 0]

    yuv[:, :, 0] = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    yuv[:, :, 1] = -0.169 * r_channel - 0.331 * g_channel + 0.499 * b_channel + 128
    yuv[:, :, 2] = 0.499 * r_channel - 0.418 * g_channel - 0.0813 * b_channel + 128

    return yuv

''' 
@cuda.jit('void(f4[:, :], f4[:], f4[:])')
def cu_matrix_vector(A, b, c):
    row = cuda.grid(1)
    if row < 3:
        sum = 0

        for i in range(3):
            sum += A[row, i] * b[i]
        
        c[row] = sum
'''
'''
@vectorize(['void(float32[:], float32[:], float32[:],'
            'float32[:], float32[:], float32[:], int32, int32)'],
            target="cuda")
def gpu_rbg2yuv(r_channel, g_channel, b_channel, y_channel, u_channel, v_channel, height, width):

    for i in prange(height * width):

        y_channel[i] = 0.299 * r_channel[i] + 0.587 * g_channel[i] + 0.114 * b_channel[i]
        u_channel[i] = -0.169 * r_channel[i] - 0.331 * g_channel[i] + 0.499 * b_channel[i] + 128
        v_channel[i] = 0.499 * r_channel[i] - 0.418 * g_channel[i] - 0.0813 * b_channel[i] + 128
'''

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_y_channel(r_channel, g_channel, b_channel):
    return 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_u_channel(r_channel, g_channel, b_channel):
    return -0.169 * r_channel - 0.331 * g_channel + 0.499 * b_channel + 128

@vectorize(['float32(float32, float32, float32)'], target='cuda')
def ret_v_channel(r_channel, g_channel, b_channel):
    return 0.499 * r_channel - 0.418 * g_channel - 0.0813 * b_channel + 128

def changeFrame(image):
    
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

def cpu_execution():
    source_dir = 'resources/'
    destination_dir = 'outputCPU/'

    list_of_files = list(os.walk(source_dir))[0][2]

    ## Timing the CPU (numpy version)
    start = timer()
    for file_name in list_of_files:
        try:
            image = cv2.imread(source_dir + file_name)
            ret_image = cpu_rbg2yuv(image)
            cv2.imwrite(destination_dir + file_name, ret_image)
        except Exception as e:
            print(f'File {file_name} not accessible')
            print(e)
    end = timer()
    print(f'Processed total {len(list_of_files)} files')
    print(f'Total time for CPU {end - start} seconds')

def gpu_execution():

    source_dir = 'resources/'
    destination_dir = 'outputGPU/'

    list_of_files = list(os.walk(source_dir))[0][2]

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

def videoCaptureGPU(file_name):

    start = timer()
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputGPU.mp4', fourcc, 20.0, (640, 480))

    while(cap.isOpened()):

        ret, frame = cap.read()

        if ret == False:
            break

        yuv = changeFrame(frame)

        out.write(yuv)

        #cv2.imshow('frame', yuv)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

    end = timer()
    print(f'Total time for GPU to convert video {end - start} seconds')
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def videoCaptureCPU(file_name):

    start = timer()
    cap = cv2.VideoCapture(file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('outputCPU.mp4', fourcc, 20.0, (640, 480))

    while(cap.isOpened()):

        ret, frame = cap.read()
        
        if ret == False:
            break

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        out.write(yuv)

        #cv2.imshow('frame', yuv)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    
    end = timer()

    print(f'Total time for CPU to convert video {end - start} seconds')
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():

    cpu_execution()
    gpu_execution()
    #videoCaptureCPU('input.mp4')
    #videoCaptureGPU('input.mp4')


    

if __name__ == '__main__':
    main()