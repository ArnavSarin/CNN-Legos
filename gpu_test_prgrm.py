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


#FEATURE LEARNING CONVOLUTION PARALLELISM

#fake_img = np.array([[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]).astype(np.float32)
#fake_filters = np.array([[[1.0,1.0],[1.0,1.0]],[[2.0,2.0],[2.0,2.0]]]).astype(np.float32)
#conv_img = np.zeros((2,2,2)).astype(np.float32)

#fake_img = np.arange(2*200*200).reshape(2,200, 200).astype(np.float32)
#fake_filters = np.array([[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],[[2.0,2.0,2.0],[2.0,2.0,2.0],[2.0,2.0,2.0]]]).astype(np.float32)
#conv_img = np.zeros((4,198,198)).astype(np.float32)
#
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







#BACKPROPOGATION FILTER CHANGE PARALLELISM


#loss_out_pt1 = np.zeros((2,2,2))
#loss_out_pt2 = np.ones((2,2,2))
#loss_out_pt3 = np.full((2, 2, 2), 2)
#loss_out = np.concatenate((loss_out_pt1, loss_out_pt2), axis=0)
#loss_out = np.concatenate((loss_out, loss_out_pt3), axis=0).astype(np.float32)
#print(loss_out)
#print()
#
#fpool = np.arange(2*4*4).reshape(2,4,4).astype(np.float32)
#fchange = np.zeros((6,3,3)).astype(np.float32)
#
#print(fpool)
#
#mf = cl.mem_flags
#cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = loss_out.flatten())
#cl_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fpool.flatten())
#cl_c = cl.Buffer(ctx, mf.WRITE_ONLY, fchange.flatten().nbytes)
#
#
#prg2 = cl.Program(ctx, """
#__kernel void conv_backprop(int lmat, int pmat, int cgmat, int lrow, int lcol, int prow, int pcol, int cgrow, int cgcol, __global float * loss_out, __global float * fpool, __global float * fchange)
#{
#
#    int i = get_global_id(0);
#    int j = get_global_id(1);
#    int k = get_global_id(2);
#
#    fchange[(i * (cgcol * cgrow)) + (j * cgcol) + k ] = 0;
#
#    for (int row=0; row < lrow; row++)
#    {
#        for (int col=0; col < lcol; col++)
#        {
#
#            /*lmat/fpool is total loss matrices / total filter matrices*/
#            fchange[(i * (cgcol * cgrow)) + (j * cgcol) + k ] += fpool[((i%pmat) * (pcol * prow)) + (row * pcol + col)+(j*pcol+k)] * loss_out[(i * (lrow * lcol)) +  (row * lcol) + col];
#
#        }
#    }
#}
#""").build()
#
#
#(lmat, pmat, cgmat, lrow, lcol, prow, pcol, cgrow, cgcol) = ( 6, 2, 6, 2, 2, 4, 4, 3, 3)
#
#lmat = np.int32(lmat)
#pmat = np.int32(pmat)
#cgmat = np.int32(cgmat)
#lrow = np.int32(lrow)
#lcol = np.int32(lcol)
#prow = np.int32(prow)
#pcol = np.int32(pcol)
#cgrow = np.int32(cgrow)
#cgcol = np.int32(cgcol)
#
#t0 = datetime.datetime.now()
#
#print(fchange.shape)
#
#prg2.conv_backprop(queue, fchange.shape ,None, lmat, pmat, cgmat, lrow, lcol, prow, pcol, cgrow, cgcol,cl_a, cl_b, cl_c)
#
#wfilter2 = np.zeros((6,3,3), dtype=np.float32)
#wfilter3 = np.empty_like(wfilter2)
#cl.enqueue_copy(queue, wfilter3 , cl_c)
#
#print(wfilter3)
#
#wfilter3 = np.split(wfilter3,int(lmat/pmat))
#wfilter3 = np.sum(wfilter3,axis=1)
#
#benchmark3 = datetime.datetime.now() - t0
#print("\nNP SPLIT")
#print(wfilter3.shape)
#print(wfilter3[0][0][0])
#print(wfilter3[1][0][0])
#print('OpenCL Multiplication: ' + str(benchmark3))



#CONVOLUTION PARALLELISM LOSS ERROR/ 1ST LAYER LOSS OUTPUT (CHANGE IN LOSS ERROR / CHANGE IN X INPUT)


#loss_out = np.arange(6*2*2).reshape(6,2,2).astype(np.float32)
#
#print("LOSS OUT BEFORE PADDING")
#print(loss_out)
#print(loss_out.shape)
#print()
#
##NEED TO PAD LOSS_OUT
#loss_out = np.pad(loss_out,((0,0),(2,2),(2,2)),constant_values=0)
#
#print("LOSS OUT AFTER PADDING")
#print(loss_out)
#print(loss_out.shape)
#print()
#
#
#new_lo_output = np.zeros((6,4,4)).astype(np.float32)
#
#ffilter_pt1 = np.zeros((1,3,3))
#ffilter_pt2 = np.ones((1,3,3))
#ffilter_pt3 = np.full((1,3,3),2)
#ffilter = np.concatenate((ffilter_pt1, ffilter_pt2), axis=0)
#ffilter = np.concatenate((ffilter, ffilter_pt3), axis=0).astype(np.float32)
#
#print("FFILTER")
#print(ffilter)
#print()
#
#
#mf = cl.mem_flags
#cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = loss_out.flatten())
#cl_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = ffilter.flatten())
#cl_c = cl.Buffer(ctx, mf.WRITE_ONLY, new_lo_output.flatten().nbytes)
#
#prg2 = cl.Program(ctx, """
#__kernel void conv_backprop_loss_o(int olmat, int fmat, int nlmat, int olrow, int olcol, int frow, int fcol, int nlrow, int nlcol, __global float * loss_out, __global float * ffilter, __global float * new_lo_output)
#{
#
#    int i = get_global_id(0);
#    int j = get_global_id(1);
#    int k = get_global_id(2);
#
#    new_lo_output[(i * (nlcol * nlrow)) + (j * nlcol) + k ] = 0;
#
#    for (int row=0; row < frow; row++)
#    {
#        for (int col=0; col < fcol; col++)
#        {
#
#            /*lmat/fpool is total loss matrices / total filter matrices*/
#            new_lo_output[(i * (nlcol * nlrow)) + (j * nlcol) + k ] += loss_out[(i * (olcol * olrow)) + (row * olcol + col)+(j*olcol+k)] * ffilter[((i/(olmat/fmat)) * (frow * fcol)) +  (row * fcol) + col];
#
#        }
#    }
#}
#""").build()
#
#(olmat, fmat, nlmat, olrow, olcol, frow, fcol, nlrow, nlcol) = ( 6, 3, 6, 6, 6, 3, 3, 4, 4)
#
#olmat = np.int32(olmat)
#fmat = np.int32(fmat)
#nlmat = np.int32(nlmat)
#olrow = np.int32(olrow)
#olcol = np.int32(olcol)
#frow = np.int32(frow)
#fcol = np.int32(fcol)
#nlrow = np.int32(nlrow)
#nlcol = np.int32(nlcol)
#
#t0 = datetime.datetime.now()
#
#prg2.conv_backprop_loss_o(queue, new_lo_output.shape ,None, olmat, fmat, nlmat, olrow, olcol, frow, fcol, nlrow, nlcol,cl_a, cl_b, cl_c)
#
#nl_output2 = np.zeros((6,4,4), dtype=np.float32)
#nl_output2 = np.empty_like(nl_output2)
#cl.enqueue_copy(queue, nl_output2 , cl_c)
#
#print("NL OUTPUT2 NO SUMMATION")
#print(nl_output2)
#print()
#
##NEED TO SUM BY INTERVALS OF 2
#nl_output2 = np.split(nl_output2,fmat)
#nl_output2 = np.sum(nl_output2,axis=0)
#
#print("SUMMED NL OUTPUT 2")
#print(nl_output2)
#benchmark5 = datetime.datetime.now() - t0
#print('OpenCL BACKPROP LOSS OUT CONVOLUTION: ' + str(benchmark5))




#POOLING PARALLELISM

pre_pool = np.arange(4*100*100).reshape(4,100,100).astype(np.float32)
aft_pool = np.zeros((4,50,50)).astype(np.float32)
pool_x = np.zeros((4,50,50)).astype(np.float32)
pool_y = np.zeros((4,50,50)).astype(np.float32)


print(pre_pool)

mf = cl.mem_flags
cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pre_pool.flatten())
cl_b = cl.Buffer(ctx, mf.WRITE_ONLY, aft_pool.flatten().nbytes)
cl_c = cl.Buffer(ctx, mf.WRITE_ONLY, pool_x.flatten().nbytes)
cl_d = cl.Buffer(ctx, mf.WRITE_ONLY, pool_y.flatten().nbytes)

prg3 = cl.Program(ctx, """
__kernel void pooling_parallelism( int pool_row, int pool_col,int prep_row, int prep_col, int aftp_row, int aftp_col,  __global float * pre_pool, __global float * aft_pool, __global float * pool_x, __global float * pool_y)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);

    aft_pool[(i * (aftp_col * aftp_row)) + (j * aftp_col) + k ] = 0;

    for (int row=0; row < pool_row; row++)
    {
        for (int col=0; col < pool_col; col++)
        {

            if( aft_pool[(i * (aftp_col * aftp_row)) + (j * aftp_col) + k ] <= pre_pool[((i * (prep_col * prep_row)) + (row * prep_col + col)+(j*pool_row*prep_col+k*pool_col))])
            {
                aft_pool[(i * (aftp_col * aftp_row)) + (j * aftp_col) + k ] = pre_pool[((i * (prep_col * prep_row)) + (row * prep_col + col)+(j*pool_row*prep_col+k*pool_col))];
                pool_y[(i * (aftp_col * aftp_row)) + (j * aftp_col) + k ] = (j*pool_row) + row;
                pool_x[(i * (aftp_col * aftp_row)) + (j * aftp_col) + k ] = (k*pool_col) + col;
            }
        }
    }
}
""").build()


(pool_row, pool_col, aftp_row, aftp_col, prep_col, prep_row) = (2,2,50,50,100,100)

pool_row = np.int32(pool_row)
pool_col = np.int32(pool_col)
aftp_row = np.int32(aftp_row)
aftp_col = np.int32(aftp_col)
prep_row = np.int32(prep_row)
prep_col = np.int32(prep_col)


t0 = datetime.datetime.now()

prg3.pooling_parallelism(queue, aft_pool.shape ,None, pool_row, pool_col, prep_row, prep_col, aftp_row, aftp_col, cl_a, cl_b, cl_c, cl_d)

aft_pool = np.zeros((4,50,50), dtype=np.float32)
aft_pool = np.empty_like(aft_pool)
cl.enqueue_copy(queue, aft_pool , cl_b)


pool_x = np.zeros((4,50,50), dtype=np.float32)
pool_x = np.empty_like(pool_x)
cl.enqueue_copy(queue, pool_x , cl_c)

pool_y = np.zeros((4,50,50), dtype=np.float32)
pool_y = np.empty_like(pool_y)
cl.enqueue_copy(queue, pool_y , cl_d)


print("\nAFTER POOL")
print(aft_pool.shape)
print(aft_pool)


benchmark4 = datetime.datetime.now() - t0
print('OpenCL Multiplication: ' + str(benchmark4))





#MAX POOLING DERIVATIVE PARALLELISM

aft_p_loss = np.arange(4*50*50).reshape(4,50,50).astype(np.float32)
pre_p_loss = np.zeros((4,100,100)).astype(np.float32)

print("\nAFTER POOLING LOSS")
print(aft_p_loss)

pool_x = pool_x.astype(np.int32)
pool_y = pool_y.astype(np.int32)

print("\nX VALUE POOL")
print(pool_x.shape)
print(pool_x)

print("\nY VALUE POOL")
print(pool_y.shape)
print(pool_y)

mf = cl.mem_flags
cl_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pool_x.flatten())
cl_b = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pool_y.flatten())
cl_c = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = aft_p_loss.flatten())
cl_d = cl.Buffer(ctx, mf.WRITE_ONLY, pre_p_loss.flatten().nbytes)


prg4 = cl.Program(ctx, """
__kernel void max_pooling_deriv_parallel(int pool_row, int pool_col, int aftprow, int aftpcol, int preprow, int prepcol,  __global int * pool_x, __global int * pool_y, __global float * aft_p_loss, __global float * pre_p_loss)
{

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);


    if( (j == pool_y[(i * aftprow * aftpcol) + ((j/pool_row)*aftpcol) + (k/pool_col)])  &&  (k == pool_x[(i * aftprow * aftpcol) + ((j/pool_row)*aftpcol) + (k/pool_col)]) )
    {
        pre_p_loss[(i * preprow * prepcol) + (j * prepcol) + k] = aft_p_loss[(i * aftprow * aftpcol) + ((j/pool_row)*aftpcol) + (k/pool_col)];
    }
    else
    {
        pre_p_loss[(i * preprow * prepcol) + (j * prepcol) + k] = 0;
    }
        
    
}
""").build()

(pool_row, pool_col, aftprow, aftpcol, prepcol, preprow) = (2,2,50,50,100,100)

pool_row = np.int32(pool_row)
pool_col = np.int32(pool_col)
aftprow = np.int32(aftprow)
aftpcol = np.int32(aftpcol)
preprow = np.int32(preprow)
prepcol = np.int32(prepcol)

t0 = datetime.datetime.now()

prg4.max_pooling_deriv_parallel(queue, pre_p_loss.shape ,None, pool_row, pool_col, aftprow, aftpcol, preprow, prepcol, cl_a, cl_b, cl_c, cl_d)

pre_p_loss = np.zeros((4,100,100), dtype=np.float32)
pre_p_loss = np.empty_like(pre_p_loss)
cl.enqueue_copy(queue, pre_p_loss , cl_d)

print("\nPRE POOLING LOSS")
print(pre_p_loss.shape)
print(pre_p_loss)
print(pre_p_loss[0][1][99])

benchmark6 = datetime.datetime.now() - t0
print('MAX POOLING DERIVATIVE: ' + str(benchmark6))
