import cpu
import gpu
import argparse

def main():

    parser = argparse.ArgumentParser(description="RGB to YUV conversion of images or video file")
    parser.add_argument('-c', help='Select CPU for processing', action='store_true')
    parser.add_argument('-g', help='Select GPU for processing', action='store_true')
    parser.add_argument('-i', action='store_true', help='Argument to convert images in RGB -> YUV')
    parser.add_argument('-v', action='store_true', help='Argument to convert video file in RGB -> YUV')
    parser.add_argument('-s', help='Source directory for images in RGB')
    parser.add_argument('-d', help='Destination directory to store converted images')
    parser.add_argument('--input', help='File path to video to be converted')
    parser.add_argument('--output', help='Output path to video after conversion')

    args = parser.parse_args()
    
    if(args.i and args.v):
        print('You can select only video or image not both')
        exit(-1)

    try:
        if(args.i == True and args.s != None and args.d != None):
            if(args.i and args.c):
                cpu.convert_images_cpu(args.s, args.d)
            if(args.i and args.g):
                gpu.convert_images_gpu(args.s, args.d)
        elif(args.i == True):
            print('Please enter correct source and destination directory')

        if(args.v == True and args.input != None and args.output != None):
            if(args.v and args.c):
                cpu.cpu_video_convert(args.input, args.output)
            if(args.v and args.g):
                gpu.gpu_video_convert(args.input, args.output)
        elif(args.v == True):
            print('Please enter correct input and output file')
    except Exception as e:
        print(e)
    

if __name__ == '__main__':
    main()