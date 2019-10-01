import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
# Transfering data from host to GPU

import numpy
a = numpy.random.randn(1000,1000)

a = a.astype(numpy.float32)

mem_copy_start_time = time.time_ns()
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)
print(f'>Took {(time.time_ns() - mem_copy_start_time) * (10 ** 9)} secs to copy array to GPU')
# Executing a kernel

gpu_exec_start_time = time.time_ns()
kernel = SourceModule("""
    __global__ void doublify(float *a){
        int idx = thredIdx.x + threadIdx.y*4;
        a[idx] *= 2;
    }
""")
func = kernel.get_function("doublify")
func(a_gpu, block=(4,4,1))
print(f'>Took {(time.time_ns() - gpu_exec_start_time) * (10 ** 9)} secs to exec on GPU')
