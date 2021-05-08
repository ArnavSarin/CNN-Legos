import pyopencl as cl
import numpy as np
import datetime
#platforms = cl.get_platforms()
#my_platform = platforms[0]
#print(my_platform.vendor)
#
#devices = my_platform.get_devices()
#my_device = devices[0]
#print(my_device.name)

#https://stackoverflow.com/questions/15235109/pyopencl-matrix-multiplication
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)


#fake_img = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]).astype(np.float32)
#fake_filters = np.array([[[1.0,1.0],[1.0,1.0]],[[2.0,2.0],[2.0,2.0]]]).astype(np.float32)
#conv_img = np.zeros((2,2,2)).astype(np.float32)

#fake_img = np.arange(2*200*200).reshape(2,200, 200).astype(np.float32)
#fake_filters = np.array([[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],[[2.0,2.0,2.0],[2.0,2.0,2.0],[2.0,2.0,2.0]]]).astype(np.float32)
#conv_img = np.zeros((4,198,198)).astype(np.float32)
#
#
##print(np.sum(fake_img[0:3,0:3]*fake_filters[0]))
##print(fake_img)
##print(fake_filters)
##print(fake_img[0:2,0:2])
##print(fake_img.flatten())
##print(fake_filters.flatten())
##print(fake_filters[0].flatten())
##print(conv_img[0].flatten())
##print(conv_img[0].flatten()[0].nbytes)
#
#mf = cl.mem_flags
#cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fake_img.flatten())
#cl_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fake_filters.flatten())
#cl_c = cl.Buffer(ctx, mf.WRITE_ONLY, conv_img.flatten().nbytes)
#
##ROWS IN FILTER, COLS IN FILTER, NUM OF MATRIX IN FILTER = frow, fcol, fmatrix
##ROWS IN CONV IMAGE, COLS IN CONV IMAGE, NUM OF MATRIX IN CONV IMAGE = crow, ccol, cmatrix_size
##ROWS IN IMAGE, COLS IN IMAGE, NUM OF MATRIX IMAGE = irow, icol, imatrix
#
#
##(index_of_matrix * matrix_size)
#
#prg = cl.Program(ctx, """
#    __kernel void multiplymatrices(int fmatrix, int cmatrix, int imatrix, int irow, int icol, int frow, int fcol, int crow, int ccol, __global float * fake_img, __global float * fake_filters, __global float * conv_img)
#    {
#
#
#        int i = get_global_id(0);
#        int j = get_global_id(1);
#        int k = get_global_id(2);
#
#        conv_img[(i * (ccol * crow)) + (j * ccol) + k ] = 0;
#
#        for (int row=0; row < frow; row++)
#        {
#            for (int col=0; col < fcol; col++)
#            {
#
#                /*(i * col + j) = translation of conv image to reg image start
#                (row * icol + col) = creating the subarray in the matrix*/
#
#                conv_img[(i * (ccol * crow)) + (j * ccol) + k] += fake_img[((i/fmatrix) * (icol * irow)) + (row * icol + col)+(j*icol+k)] * fake_filters[((i%fmatrix) * (fcol * frow)) + (row * frow) + col];
#
#            }
#        }
#
#    }
#    """).build()
#
#
##print(conv_img[0].shape)
#
#(fmatrix_size, cmatrix_size, imatrix_size, irow, icol, frow, fcol, crow, ccol) = ( 2, 4, 2, 200, 200, 3, 3, 198, 198)
#
#fmatrix_size = np.int32(fmatrix_size)
#cmatrix_size = np.int32(cmatrix_size)
#imatrix_size = np.int32(imatrix_size)
#irow = np.int32(irow)
#icol = np.int32(icol)
#frow = np.int32(frow)
#fcol = np.int32(fcol)
#crow = np.int32(crow)
#ccol = np.int32(ccol)
#
#
#t0 = datetime.datetime.now()
#
#print(conv_img.shape)
#
#prg.multiplymatrices(queue, conv_img.shape ,None, fmatrix_size, cmatrix_size, imatrix_size, irow, icol, frow, fcol, crow, ccol,cl_a, cl_b, cl_c)
#
#conv_img2 = np.zeros((4,198,198), dtype=np.float32)
#conv_img3 = np.empty_like(conv_img2)
#cl.enqueue_copy(queue, conv_img3 , cl_c)
#
#
#benchmark1 = datetime.datetime.now() - t0
#
#print(conv_img3)
#print(conv_img3[0][0][0])
#print(conv_img3[1][0][0])
#print('OpenCL Multiplication: ' + str(benchmark1))



#clength = 198
#cheight = 198
#filter_size = 2
#image_num = 2
#klength = 3
#kheight = 3
#
#convolution_image = np.zeros((4,198,198),dtype=np.float32)
#
#t1 = datetime.datetime.now()
#
#for f1 in range(filter_size * image_num):
#    for l in range(clength):
#        for h in range(cheight):
#            filtered_subset = fake_img[int(f1/filter_size),h:h+kheight,l:l+klength] * fake_filters[int(f1%filter_size)]
#            convolution_image[f1,h,l] = np.sum(filtered_subset)
#
#benchmark2 = datetime.datetime.now() - t1
#
#print(convolution_image)
#print('Original Convolution: ' + str(benchmark2))
#
#print(np.array_equal(convolution_image, conv_img3, equal_nan=True))

loss_out_pt1 = np.ones((2,2,2))
loss_out_pt2 = np.full((2, 2, 2), 2)
loss_out = np.concatenate((loss_out_pt1, loss_out_pt2), axis=0).astype(np.float32)
print(loss_out)
print()

fpool = np.arange(2*4*4).reshape(2,4,4).astype(np.float32)
fchange = np.zeros((4,3,3)).astype(np.float32)

print(fpool)

mf = cl.mem_flags
cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = loss_out.flatten())
cl_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fpool.flatten())
cl_c = cl.Buffer(ctx, mf.WRITE_ONLY, fchange.flatten().nbytes)


prg2 = cl.Program(ctx, """
__kernel void conv_backprop(int lmat, int pmat, int cgmat, int lrow, int lcol, int prow, int pcol, int cgrow, int cgcol, __global float * loss_out, __global float * fpool, __global float * fchange)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);
    
    fchange[(i * (cgcol * cgrow)) + (j * cgcol) + k ] = 0;
    
    for (int row=0; row < lrow; row++)
    {
        for (int col=0; col < lcol; col++)
        {
                
            /*lmat/fpool is total loss matrices / total filter matrices*/
            fchange[(i * (cgcol * cgrow)) + (j * cgcol) + k ] += fpool[((i%pmat) * (pcol * prow)) + (row * pcol + col)+(j*pcol+k)] * loss_out[(i * (lrow * lcol)) +  (row * lcol) + col];
        
        }
    }
}
""").build()


(lmat, pmat, cgmat, lrow, lcol, prow, pcol, cgrow, cgcol) = ( 4, 2, 4, 2, 2, 4, 4, 3, 3)

lmat = np.int32(lmat)
pmat = np.int32(pmat)
cgmat = np.int32(cgmat)
lrow = np.int32(lrow)
lcol = np.int32(lcol)
prow = np.int32(prow)
pcol = np.int32(pcol)
cgrow = np.int32(cgrow)
cgcol = np.int32(cgcol)

t0 = datetime.datetime.now()

print(fchange.shape)

prg2.conv_backprop(queue, fchange.shape ,None, lmat, pmat, cgmat, lrow, lcol, prow, pcol, cgrow, cgcol,cl_a, cl_b, cl_c)

wfilter2 = np.zeros((4,3,3), dtype=np.float32)
wfilter3 = np.empty_like(wfilter2)
cl.enqueue_copy(queue, wfilter3 , cl_c)

print(wfilter3)

wfilter3 = np.split(wfilter3,pmat)
wfilter3 = np.sum(wfilter3,axis=1)

benchmark3 = datetime.datetime.now() - t0
print("\nNP SPLIT")
print(wfilter3)
print(wfilter3[0][0][0])
print(wfilter3[1][0][0])
print('OpenCL Multiplication: ' + str(benchmark3))


