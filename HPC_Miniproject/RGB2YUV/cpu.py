import cv2
from timeit import default_timer as timer
import numpy as np
import os

def cpu_convert_frame(image):
    
    yuv = np.zeros_like(image)
    
    r_channel = image[:, :, 2]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 0]

    yuv[:, :, 0] = 0.299 * r_channel + 0.587 * g_channel + 0.114 * b_channel
    yuv[:, :, 1] = -0.169 * r_channel - 0.331 * g_channel + 0.499 * b_channel + 128
    yuv[:, :, 2] = 0.499 * r_channel - 0.418 * g_channel - 0.0813 * b_channel + 128

    return yuv

def convert_images_cpu(source_dir, destination_dir):
    #source_dir = 'resources/'
    #destination_dir = 'outputCPU/'

    list_of_files = list(os.walk(source_dir))[0][2]

    if(len(list_of_files) == 0):
        raise Exception('Files not found. Verify source and destination directory entry')

    ## Timing the CPU (numpy version)
    start = timer()
    for file_name in list_of_files:
        try:
            image = cv2.imread(source_dir + file_name)
            ret_image = cpu_convert_frame(image)
            cv2.imwrite(destination_dir + file_name, ret_image)
        except Exception as e:
            print(f'File {file_name} not accessible')
            print(e)
    end = timer()
    print(f'Processed total {len(list_of_files)} files')
    print(f'Total time for CPU {end - start} seconds')


def cpu_video_convert(in_file_name, out_file_name):

    start = timer()
    cap = cv2.VideoCapture(in_file_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_file_name, fourcc, 20.0, (640, 480))

    while(cap.isOpened()):

        ret, frame = cap.read()
        
        if ret == False:
            break

        yuv = cpu_convert_frame(frame)
        
        out.write(yuv)

        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    
    end = timer()

    print(f'Total time for CPU to convert video {end - start} seconds')
    cap.release()
    out.release()
    cv2.destroyAllWindows()