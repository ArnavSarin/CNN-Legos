import numpy as np
import math
import matplotlib.pyplot as plt
import cv2
import glob
import os
import openpyxl
import pyopencl as cl


class Convolutional_Neural_Network:

    #SOFTMAX ACTIVATION FUNCTION
    def softmax(self,matrix):
        shiftx = matrix - np.max(matrix)
        exps = np.exp(shiftx)
        return exps / np.sum(exps)

    #SOFTMAX DERIVATIVE
    def d_softmax(self,matrix):
        Sz = self.softmax(matrix)
        D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
        return D
        

    #ReLU ACTIVATION FUNCTION
    def ReLU(self,matrix):
        return matrix * (matrix > 0)
    
    #ReLU DERIVATIVE
    def d_ReLU(self,matrix):
        return 1 * (matrix > 0)
            
    #SIGMOID ACTIVATION FUNCTION
    def sigmoid(self,matrix):
#        z = 1/(1 + np.exp(-matrix))
#        return z
        return np.tanh(matrix)
    
    #SIGMOID DERIVATIVE
    def d_sigmoid(self,matrix):
#        f = 1/(1 + np.exp(-matrix))
#        df = f * (1 - f)
#        return df
        dt=1-(self.sigmoid(matrix)**2)
        return dt

    #MAX POOLING
    def max_pooling(self,matrix):
        return np.amax(matrix)

    #MAX POOLING DERIVATIVE
    def dmax_pooling(self,before_pooling, after_pooling, error):
    
        #RUN THROUGH EACH CONVOLUTION
        for f in range (after_pooling.shape[0]):
            #RUN THROUGH THE LENGTH OF MATRIX
            for l in range (after_pooling.shape[2]):
                #RUN THROUGH HEIGHT OF MATRIX
                for h in range (after_pooling.shape[1]):
                    h_iter = h * self.pool_height
                    l_iter = l * self.pool_length
                    before_pooling[f,h_iter:h_iter+self.pool_height,l_iter:l_iter+self.pool_length] == np.where(after_pooling[f,h,l] == before_pooling[f,h_iter:h_iter+self.pool_height,l_iter:l_iter+self.pool_length],1 * error[f,h,l],0)
        
        return before_pooling


    #AVERAGE POOLING
    def avg_pooling(self,matrix):
        return (np.sum(matrix)) / matrix.size
        
    
    #CONVOLUTION PARALLELISM
    def conv_parallelism(self,fmatrix, imatrix, cmatrix):
    
        #CONVERT MATRIX TO FLOAT32
        imatrix = imatrix.astype(np.float32)
        fmatrix = fmatrix.astype(np.float32)
        cmatrix = cmatrix.astype(np.float32)
        
        #CREATE BUFFER FOR EACH ARRAY
        mf = cl.mem_flags
        cl_a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = imatrix.flatten())
        cl_b = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = fmatrix.flatten())
        cl_c = cl.Buffer(self.ctx, mf.WRITE_ONLY, cmatrix.flatten().nbytes)
        
        #PARALLELISM PROGRAM IN C
        prg = cl.Program(self.ctx, """
        __kernel void multiplymatrices(int fmatrix, int cmatrix, int imatrix, int irow, int icol, int frow, int fcol, int crow, int ccol, __global float * fake_img, __global float * fake_filters, __global float * conv_img)
        {
        
            int i = get_global_id(0);
            int j = get_global_id(1);
            int k = get_global_id(2);
            
            conv_img[(i * (ccol * crow)) + (j * ccol) + k ] = 0;

            for (int row=0; row < frow; row++)
            {
                for (int col=0; col < fcol; col++)
                {
            
                    /*(i * col + j) = translation of conv image to reg image start
                    (row * icol + col) = creating the subarray in the matrix*/
                    
                    conv_img[(i * (ccol * crow)) + (j * ccol) + k] += fake_img[((i/fmatrix) * (icol * irow)) + (row * icol + col)+(j*icol+k)] * fake_filters[((i%fmatrix) * (fcol * frow)) + (row * fcol) + col];
                
                }
            }
            
        }
        """).build()
        
        #CONVERTING FMATRIX DIMENSIONS TO INT32
        if(fmatrix.ndim == 3):
            fmatrix_size = np.int32(fmatrix.shape[0])
            frow = np.int32(fmatrix.shape[1])
            fcol = np.int32(fmatrix.shape[2])
        else:
            fmatrix_size = np.int32(1)
            frow = np.int32(fmatrix.shape[0])
            fcol = np.int32(fmatrix.shape[1])
        
        
        #CONVERTING IMATRIX DIMENSIONS TO INT32
        if(imatrix.ndim == 3):
            imatrix_size = np.int32(imatrix.shape[0])
            irow = np.int32(imatrix.shape[1])
            icol = np.int32(imatrix.shape[2])
        else:
            imatrix_size = np.int32(1)
            irow = np.int32(imatrix.shape[0])
            icol = np.int32(imatrix.shape[1])
        
        
        #CONVERTING CMATRIX DIMENSIONS TO INT32
        if(cmatrix.ndim == 3):
            cmatrix_size = np.int32(cmatrix.shape[0])
            crow = np.int32(cmatrix.shape[1])
            ccol = np.int32(cmatrix.shape[2])
        else:
            cmatrix_size = np.int32(1)
            crow = np.int32(cmatrix.shape[0])
            ccol = np.int32(cmatrix.shape[1])
        
        
        #RUN PARALLELISM ALGORITHM IN C
        prg.multiplymatrices(self.queue, cmatrix.shape ,None, fmatrix_size, cmatrix_size, imatrix_size, irow, icol, frow, fcol, crow, ccol,cl_a, cl_b, cl_c)
        
        #CREATE ARRAY OF CONV IMAGE FILLED WITH ZEROS
        cimg = np.zeros_like(cmatrix,dtype=np.float32)
        cimg = np.empty_like(cimg)
        
        #COPY OVER PARALLELISM ALGORITHM RESULT TO CONV IMAGE ARRAY
        cl.enqueue_copy(self.queue, cimg , cl_c)
        
        return cimg
        
    
    
    def backprop_conv(self, loss_output, pooling_lay, weight_change):
        
        #CONVERT MATRIX TO FLOAT32
        loss_output = loss_output.astype(np.float32)
        pooling_lay = pooling_lay.astype(np.float32)
        weight_change = weight_change.astype(np.float32)
        
        
        #CREATE BUFFER FOR EACH ARRAY
        mf = cl.mem_flags
        cl_a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = loss_output.flatten())
        cl_b = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pooling_lay.flatten())
        cl_c = cl.Buffer(self.ctx, mf.WRITE_ONLY, weight_change.flatten().nbytes)
        
        
        prg2 = cl.Program(self.ctx, """
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
        
        #CONVERTING LOSS CHANGE/OUTPUT CHANGE DIMENSIONS TO INT32
        if(loss_output.ndim == 3):
            loss_output_size = np.int32(loss_output.shape[0])
            lrow = np.int32(loss_output.shape[1])
            lcol = np.int32(loss_output.shape[2])
        else:
            loss_output_size = np.int32(1)
            lrow = np.int32(loss_output.shape[0])
            lcol = np.int32(loss_output.shape[1])
        
        
        #CONVERTING POOLING LAYER DIMENSIONS TO INT32
        if(pooling_lay.ndim == 3):
            pooling_lay_size = np.int32(pooling_lay.shape[0])
            prow = np.int32(pooling_lay.shape[1])
            pcol = np.int32(pooling_lay.shape[2])
        else:
            pooling_lay_size = np.int32(1)
            prow = np.int32(pooling_lay.shape[0])
            pcol = np.int32(pooling_lay.shape[1])
        
        
        #CONVERTING FILTER WEIGHT CHANGE DIMENSIONS TO INT32
        if(weight_change.ndim == 3):
            weight_change_size = np.int32(weight_change.shape[0])
            cgrow = np.int32(weight_change.shape[1])
            cgcol = np.int32(weight_change.shape[2])
        else:
            weight_change_size = np.int32(1)
            cgrow = np.int32(weight_change.shape[0])
            cgcol = np.int32(weight_change.shape[1])
            
    
        #RUN PARALLELISM ALGORITHM IN C
        prg2.conv_backprop(self.queue, weight_change.shape ,None, loss_output_size, pooling_lay_size, weight_change_size, lrow, lcol, prow, pcol, cgrow, cgcol,cl_a, cl_b, cl_c)
    
    
        #CREATE ARRAY OF CONV IMAGE FILLED WITH ZEROS
        wchange = np.zeros_like(weight_change,dtype=np.float32)
        wchange = np.empty_like(wchange)
        
        #COPY OVER PARALLELISM ALGORITHM RESULT TO CONV IMAGE ARRAY
        cl.enqueue_copy(self.queue, wchange , cl_c)
        
        #SUMMING THE WEIGHT CHANGE TO HAVE SAME DIMENSIONS AS THE FILTERS
        wchange = np.split(wchange,int(loss_output_size/pooling_lay_size))
        wchange = np.sum(wchange,axis=1)
        
        return wchange
    
    
    
    def parallel_pooling(self, pre_pool, aft_pool):
    
        #CONVERT MATRIX TO FLOAT32
        pre_pool = pre_pool.astype(np.float32)
        aft_pool = aft_pool.astype(np.float32)
        
        mf = cl.mem_flags
        cl_a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = pre_pool.flatten())
        cl_b = cl.Buffer(self.ctx, mf.WRITE_ONLY, aft_pool.flatten().nbytes)
        
        prg3 = cl.Program(self.ctx, """
        __kernel void pooling_parallelism( int pool_row, int pool_col,int prep_row, int prep_col, int aftp_row, int aftp_col,  __global float * pre_pool, __global float * aft_pool)
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
                    }
                }
            }
        }
        """).build()
        
        
        #CONVERTING PRE POOLING LAYER DIMENSIONS TO INT32
        if(pre_pool.ndim == 3):
            pre_pool_size = np.int32(pre_pool.shape[0])
            prep_row = np.int32(pre_pool.shape[1])
            prep_col = np.int32(pre_pool.shape[2])
        else:
            pooling_lay_size = np.int32(1)
            prep_row = np.int32(pre_pool.shape[0])
            prep_col = np.int32(pre_pool.shape[1])
        
        
        #CONVERTING AFTER POOLING LAYER DIMENSIONS TO INT32
        if(aft_pool.ndim == 3):
            aft_pool_size = np.int32(aft_pool.shape[0])
            aftp_row = np.int32(aft_pool.shape[1])
            aftp_col = np.int32(aft_pool.shape[2])
        else:
            aft_pool_size = np.int32(1)
            aftp_row = np.int32(aft_pool.shape[0])
            aftp_col = np.int32(aft_pool.shape[1])
            
            
            
        #RUN PARALLELISM ALGORITHM IN C
        prg3.pooling_parallelism(self.queue, aft_pool.shape ,None, np.int32(self.pool_length), np.int32(self.pool_height), prep_row, prep_col, aftp_row, aftp_col, cl_a, cl_b)
    
        #COPY OVER PARALLELISM ALGORITHM RESULT TO CONV IMAGE ARRAY
        cl.enqueue_copy(self.queue, aft_pool , cl_b)
        
        
        return aft_pool
    
        
    def conv_backprop_loss(self, oloss_out, ofilter, new_lo_output):
        
        #CONVERT MATRIX TO FLOAT32
        oloss_out = oloss_out.astype(np.float32)
        ofilter = ofilter.astype(np.float32)
        new_lo_output = new_lo_output.astype(np.float32)
        
        #CREATE BUFFER FOR EACH ARRAY
        mf = cl.mem_flags
        cl_a = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = oloss_out.flatten())
        cl_b = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = ofilter.flatten())
        cl_c = cl.Buffer(self.ctx, mf.WRITE_ONLY, new_lo_output.flatten().nbytes)
        
        
        prg4 = cl.Program(self.ctx, """
        __kernel void conv_backprop_loss_o(int olmat, int fmat, int nlmat, int olrow, int olcol, int frow, int fcol, int nlrow, int nlcol, __global float * loss_out, __global float * ffilter, __global float * new_lo_output)
        {

            int i = get_global_id(0);
            int j = get_global_id(1);
            int k = get_global_id(2);

            new_lo_output[(i * (nlcol * nlrow)) + (j * nlcol) + k ] = 0;

            for (int row=0; row < frow; row++)
            {
                for (int col=0; col < fcol; col++)
                {

                    /*lmat/fpool is total loss matrices / total filter matrices*/
                    new_lo_output[(i * (nlcol * nlrow)) + (j * nlcol) + k ] += loss_out[(i * (olcol * olrow)) + (row * olcol + col)+(j*olcol+k)] * ffilter[((i/(olmat/fmat)) * (frow * fcol)) +  (row * fcol) + col];

                }
            }
        }
        """).build()
        
        
        #CONVERTING ORIGINAL LOSS CHANGE/OUTPUT DIMENSIONS TO INT32
        if(oloss_out.ndim == 3):
            oloss_out_size = np.int32(oloss_out.shape[0])
            olrow = np.int32(oloss_out.shape[1])
            olcol = np.int32(oloss_out.shape[2])
        else:
            oloss_out_size = np.int32(1)
            olrow = np.int32(oloss_out.shape[0])
            olcol = np.int32(oloss_out.shape[1])
               
               
        #CONVERTING ORIGINAL FILTER DIMENSIONS TO INT32
        if(ofilter.ndim == 3):
            ofilter_size = np.int32(ofilter.shape[0])
            frow = np.int32(ofilter.shape[1])
            fcol = np.int32(ofilter.shape[2])
        else:
            ofilter_size = np.int32(1)
            frow = np.int32(ofilter.shape[0])
            fcol = np.int32(ofilter.shape[1])
               
               
        #CONVERTING NEW LOSS CHANGE/OUTPUT DIMENSIONS TO INT32
        if(new_lo_output.ndim == 3):
            new_lo_output_size = np.int32(new_lo_output.shape[0])
            nlrow = np.int32(new_lo_output.shape[1])
            nlcol = np.int32(new_lo_output.shape[2])
        else:
            new_lo_output_size = np.int32(1)
            nlrow = np.int32(new_lo_output.shape[0])
            nlcol = np.int32(new_lo_output.shape[1])
            
        #RUN PARALLELISM ALGORITHM IN C
        prg4.conv_backprop_loss_o(self.queue, new_lo_output.shape ,None, oloss_out_size, ofilter_size, new_lo_output_size, olrow, olcol, frow, fcol, nlrow, nlcol,cl_a, cl_b, cl_c)
        
        #CREATE ARRAY OF CONV IMAGE FILLED WITH ZEROS
        nl_output = np.zeros_like(new_lo_output,dtype=np.float32)
        nl_output = np.empty_like(nl_output)
        
        #COPY OVER PARALLELISM ALGORITHM RESULT TO CONV IMAGE ARRAY
        cl.enqueue_copy(self.queue, nl_output , cl_c)
        
        #SUMMING ALL FILTER LOSSES FOR EACH ORIGINAL IMAGE
        nl_output2 = np.split(nl_output,ofilter_size)
        nl_output2 = np.sum(nl_output2,axis=0)
        
        return nl_output2
        
        
        

    def __init__(self):
        
        #FILTER DIMENSIONS (KERNEL)
        self.kernel_height = 3
        self.kernel_length = 3
        
        #NUMBER OF FILTERS PER CONVOLUTION LAYER
        self.num_filtersL1 = 2
        self.num_filtersL2 = 3
        
        #LAYER 1 KERNEL INITIALIZATION
        self.filterL1 = np.random.rand(self.num_filtersL1, self.kernel_height, self.kernel_length)
        
        #LAYER 2 KERNEL INITIALIZATION
        self.filterL2 = np.random.rand(self.num_filtersL2, self.kernel_height, self.kernel_length)
    
        #POOLING DIMENSIONS
        self.pool_height = 2
        self.pool_length = 2
    
        #INPUT DIMENSIONS
        self.input_height = 50
        self.input_length = 50
        
        #STRIDE
        self.stride_horizontal = 1
        self.stride_vertical = 1
        
        #INPUT SIZE OF FULLY CONNECTED LAYER OR FLATTENED LAYER
        height = (((self.input_height - self.kernel_height + 1)/ self.pool_height) - self.kernel_height + 1)/ self.pool_height
        length = (((self.input_length - self.kernel_length + 1)/self.pool_length) - self.kernel_length + 1)/ self.pool_length
        
        self.finput_neurons = height * length * self.num_filtersL1 * self.num_filtersL2
        
        
        #FULLY CONNECTED LAYER HIDDEN LAYER 1
        self.hiddenL1_neurons = 50
        
        #WEIGHTS FOR FULLY CONNECTED LAYER 1
        self.w1 = np.random.rand(int(self.finput_neurons),self.hiddenL1_neurons)
        self.w1 = self.w1 * (math.sqrt(1/self.finput_neurons))
        
        #BIAS FOR FULLY CONNECTED LAYER 1
        self.b1 = np.zeros(self.hiddenL1_neurons)
        
        #OUTPUT NEURONS
        self.o_neurons = 10
        
        #WEIGHTS FOR HIDDEN LAYER 1 TO OUTPUT
        self.w2 = np.random.rand(self.hiddenL1_neurons,self.o_neurons)
        self.w2 = self.w2 * (math.sqrt(1/self.hiddenL1_neurons))
        
        #BIAS FOR HIDDEN LAYER 1 TO OUTPUT
        self.b2 = np.zeros(self.o_neurons)
        
        #SET GPU FOR PARALLELISM
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)

        
        
        
    #FORWARD PASS
    def Feature_Learning(self,input_img):
    
    
    #LAYER 1 OF FEATURE LEARNING
    
    
        #SETTING THE INPUT IMG
        self.input_img = input_img
        
        #CALCULATE THE VALUE OF CONVOLUTION LAYER 1
        self.C1_height = self.input_height - self.kernel_height + 1
        self.C1_length = self.input_length - self.kernel_length + 1
        self.C1 = np.zeros((self.num_filtersL1, self.C1_height, self.C1_length))
        
        #CONVOLUTION FUNCTION PARALLELISM
        self.C1 = self.conv_parallelism(self.filterL1,input_img,self.C1)

        #ReLU ACTIVATION FUNCTION
        self.activ_C1 = self.ReLU(self.C1)

        #CALCULATE THE VALUE OF POOLING LAYER 1
        self.P1_height = int(self.C1_height/ self.pool_height)
        self.P1_length = int(self.C1_length/ self.pool_length)
        self.P1 = np.zeros((self.num_filtersL1, self.P1_height, self.P1_length))
        
        #RUN THROUGH EACH CONVOLUTION FOR LAYER 1
        self.P1 = self.parallel_pooling(self.activ_C1,self.P1)
        
        
        
    #LAYER 2 OF FEATURE LEARNING
    
    
        #CALCULATE THE VALUE OF CONVOLUTION LAYER 2
        self.C2_height = int(self.P1_height - self.kernel_height + 1)
        self.C2_length = int(self.P1_length - self.kernel_length + 1)
        self.C2 = np.zeros((self.num_filtersL1 * self.num_filtersL2, self.C2_height, self.C2_length))
        
        #CONVOLUTION FUNCTION PARALLELISM
        self.C2 = self.conv_parallelism(self.filterL2,self.P1,self.C2)

        #ReLU ACTIVATION FUNCTION
        self.activ_C2 = self.ReLU(self.C2)
            
        #CALCULATE THE VALUE OF POOLING LAYER 2
        self.P2_height = int(self.C2_height/self.pool_height)
        self.P2_length = int(self.C2_length/self.pool_length)
        self.P2 = np.zeros((self.num_filtersL1 * self.num_filtersL2, self.P2_height, self.P2_length))
        
        #RUN THROUGH EACH CONVOLUTION FOR LAYER 2
        self.P2 = self.parallel_pooling(self.activ_C2,self.P2)
        
    
    #FLATTEN LAYER CREATION
    
        self.flatten_layer = self.P2.flatten()
        
        return self.flatten_layer
        
    
    
    #FORWARD PASS
    def Classification(self):
        
        #CALCULATE HIDDEN LAYER 1
        self.h1_nobias = np.dot(self.flatten_layer,self.w1)
        self.h1_bias = self.h1_nobias + self.b1
        self.h1 = self.sigmoid(self.h1_bias)
        
        #CALCULATE OUTPUT LAYER
        self.z1_nobias = np.dot(self.h1,self.w2)
        self.z1_bias = self.z1_nobias + self.b2
        self.o1 = self.softmax(self.z1_bias)
        
        
        
        
    #BACKWARD PASS
    def Backpropogation(self,cost_vec,learning_rate):
        
        
    #OUTPUT - HIDDEN LAYER 1
    
    
        #CROSS ENTROPY LOSS
        error = -1 * (cost_vec/self.o1)
    
        #SOFTMAX DERIVATION
        dsoftmax = self.d_softmax(self.z1_bias)
        
        #WEIGHT CHANGE CALCULATION (JACOBIAN MULTIPLICATION)
        m_jacobian = (error * dsoftmax)
        sum_jacobian = np.sum(m_jacobian, axis=1)
        self.c_w2 = np.dot(self.h1.reshape(self.hiddenL1_neurons,1),sum_jacobian.reshape(1,self.o_neurons))
        
        #BIAS CHANGE CALCULATION
        self.c_b2 = sum_jacobian
        
        
        
    #HIDDEN LAYER 1 - FULLY CONNECTED LAYER INPUT
    
    
        #SIGMOID DERIVATION
        dsigmoid = self.d_sigmoid(self.h1)
        
        #ERROR CALCULATION
        fin_error = sum_jacobian * self.w2
        fin_error = np.sum(fin_error, axis=1)
        
        #WEIGHT CHANGE CALCULATION
        #(DERIV OF COST/ DERIV OF ACTIVATION OUTPUT) * (DERIV OF ACTIVATION OUTPUT / DERIV OF ACTIVATION INPUT)
        dcost_actinput = fin_error * dsigmoid
        self.c_w1 = np.dot(self.flatten_layer.reshape(int(self.finput_neurons),1),dcost_actinput.reshape(1,self.hiddenL1_neurons))
        
        #BIAS CHANGE CALCULATION
        self.c_b1 = dcost_actinput
        
    
    
    #BACKWARD PASS FEATURE LEARNING
    
    
        #ERROR CALCULATION
        bp_error = fin_error * self.w1
        bp_error = np.sum(bp_error,axis = 1)
        bp_error = bp_error.reshape(self.num_filtersL1 * self.num_filtersL2, self.P2_height, self.P2_length)
        
        #DERIVATIVE OF POOLING LAYER 2 AND ERROR MULTIPLICATION
        dpooling2 = self.dmax_pooling(self.activ_C2,self.P2,bp_error)
        
        #DERIVATIVE OF RELU
        dRelU_2 = self.d_ReLU(self.activ_C2)
        
        #DERIVATIVE LOSS TO OUTPUT
        loss2_out = dpooling2 * dRelU_2
        
        #FILTER 2 CHANGE INITIALIZATION
        self.c_filter2 = np.zeros((self.num_filtersL1 * self.num_filtersL2, self.kernel_height, self.kernel_length))
        
        #CONVOLUTION FUNCTION PARALLELISM
        self.c_filter2 = self.backprop_conv(loss2_out,self.P1,self.c_filter2)
        
        #NEED TO PAD LOSS
        padded_loss2 = np.pad(loss2_out,((0,0),(self.kernel_length-1,self.kernel_length-1),(self.kernel_length-1,self.kernel_length-1)),constant_values=0)
        
        #ERROR BEFORE CONVOLUTION 2
        C2_error = np.zeros((self.num_filtersL1*self.num_filtersL2,self.P1_height,self.P1_length))
        
        #ROTATE FILTER
        rotated_filterL2 = np.rot90(self.filterL2,2,(1,2))
        
        #CONVOLUTION FUNCTION PARALLELISM
        C2_error = self.conv_backprop_loss(padded_loss2,self.filterL2,C2_error)
        
        #DERIVATIVE OF POOLING LAYER 2 AND ERROR MULTIPLICATION
        dpooling = self.dmax_pooling(self.activ_C1,self.P1,C2_error)
        
        #DERIVATIVE OF RELU
        dRelU = self.d_ReLU(self.activ_C1)
        
        #DERIVATIVE LOSS TO OUTPUT
        loss_out = dpooling * dRelU
        
        #FILTER 2 CHANGE INITIALIZATION
        self.c_filter = np.zeros((self.num_filtersL1, self.kernel_height, self.kernel_length))
        
        #CONVOLUTION FUNCTION PARALLELISM
        self.c_filter = self.backprop_conv(loss_out,self.input_img,self.c_filter)


    
    #SETUP OF CLASSIFICATION WEIGHTS/FILTERS
     
     
        #CLASSIFICATION WEIGHT CALCULATION
        self.w1 = self.w1 - (learning_rate * self.c_w1)
        self.w2 = self.w2 - (learning_rate * self.c_w2)
        
        #CLASSIFICATION BIAS CALCULATION
        self.b1 = self.b1 - (learning_rate * self.c_b1)
        self.b2 = self.b2 - (learning_rate * self.c_b2)
        
        #FEATURE LEARNING FILTER CALCULATION
        self.filterL1 = self.filterL1 - (learning_rate * self.c_filter)
        self.filterL2 = self.filterL2 - (learning_rate * self.c_filter2)
        
        conv_filter_stack =  np.hstack((self.filterL1.flatten(),self.filterL2.flatten()))
        layer_one = np.hstack((self.w1.flatten(),self.b1))
        layer_two = np.hstack((self.w2.flatten(),self.b2))
        
        part1 = np.hstack((conv_filter_stack,layer_one))
        all_parts = np.hstack((part1,layer_two))
        
        self.changes = all_parts
        
        return self.changes
        
    
    
    def label(self,name):
        #LABELING EXPECTED OUTPUT VECTOR FOR BACKPROPOGATION
        self.final_expected_result = np.zeros(10)
        
        if("halfbush-" in name):
            self.final_expected_result[9] = 1.0
        elif("lever-" in name):
            self.final_expected_result[8] = 1.0
        elif("peg2-" in name):
            self.final_expected_result[7] = 1.0
        elif("rooftile-" in name):
            self.final_expected_result[6] = 1.0
        elif("1x1plate-" in name):
            self.final_expected_result[5] = 1.0
        elif("1x2plate-" in name):
            self.final_expected_result[4] = 1.0
        elif("2x2plate-" in name):
            self.final_expected_result[3] = 1.0
        elif("1x1-" in name):
            self.final_expected_result[2] = 1.0
        elif("1x2-" in name):
            self.final_expected_result[1] = 1.0
        elif("2x2-" in name):
            self.final_expected_result[0] = 1.0
            
        return self.final_expected_result
        
        
    def determine_label(self,output):
            
        #DETERMINE INDEX OF LABEL
        index_of_label = np.argmax(output)
            
        if(index_of_label == 0):
            return "2x2-"
        elif(index_of_label == 1):
            return "1x2-"
        elif(index_of_label == 2):
            return "1x1-"
        elif(index_of_label == 3):
            return "2x2plate-"
        elif(index_of_label == 4):
            return "1x2plate-"
        elif(index_of_label == 5):
            return "1x1plate-"
        elif(index_of_label == 6):
            return "rooftile-"
        elif(index_of_label == 7):
            return "peg2-"
        elif(index_of_label == 8):
            return "lever-"
        elif(index_of_label == 9):
            return "halfbush-"
        
        
    def save_parameters(self,parameters, LEGO_BLOCKS):
       
           #TO SAVE THE PARAMETERS
           np.save('parameters.npy', parameters)
           
           #TO SAVE LEGO_BLOCKS PICTURE BATCH
           np.save('lego_blocks.npy', LEGO_BLOCKS)
        
        
    def load_parameters(self):

        try:
            #LOAD WEIGHTS AND BIAS
            self.filt_weights_bias_all = np.load('parameters.npy',allow_pickle=True)
            
            #LOAD LEGO_BLOCKS BATCH
            LEGO_BLOCKS = np.load('lego_blocks.npy',allow_pickle=True)

            #SET WEIGHTS AND BIAS
            filterL1_range = self.kernel_height*self.kernel_length*self.num_filtersL1
            self.filterL1 = self.filt_weights_bias_all[0: filterL1_range].reshape(self.num_filtersL1,self.kernel_height,self.kernel_length)
            
            filterL2_range = filterL1_range + (self.kernel_height*self.kernel_length*self.num_filtersL2)
            self.filterL2 = self.filt_weights_bias_all[filterL1_range:filterL2_range].reshape(self.num_filtersL2,self.kernel_height,self.kernel_length)
            
            w1_range = int(filterL2_range + (self.finput_neurons * self.hiddenL1_neurons))
            self.w1 = self.filt_weights_bias_all[filterL2_range:w1_range].reshape(int(self.finput_neurons),self.hiddenL1_neurons)
            
            b1_range = int(w1_range + self.hiddenL1_neurons)
            self.b1 = self.filt_weights_bias_all[w1_range:b1_range]
            
            w2_range = int(b1_range + (self.hiddenL1_neurons * self.o_neurons))
            self.w2 = self.filt_weights_bias_all[b1_range:w2_range].reshape(self.hiddenL1_neurons,self.o_neurons)
            
            b2_range = int(w2_range + self.o_neurons)
            self.b2 = self.filt_weights_bias_all[w2_range:b2_range]
            
            
            #RETURN BATCH
            return LEGO_BLOCKS.tolist()
            
        except OSError as e:

            print("LOADED FILE DOESNT EXIST")
            
            #RETURN BATCH
            return {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
                
            
        
    def delete_parameters(self, file_path):

        try:
            #DELETE PARAMETERS FILE
            os.remove('parameters.npy')
        
            #DELETE LEGO_BLOCKS FILE
            os.remove('lego_blocks.npy')
        
            #DELETE EXTRA FILE
#            os.rmdir(file_path)
        
        except OSError as e:
            print("FILE DOESNT EXIST FOR DELETION")
            
        

        
def main():

    LEGO_BLOCKS = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
    
    end = False
    
    CNN = Convolutional_Neural_Network()
    
    while(not end):
    
        #TO LOAD PARAMETERS
        LEGO_BLOCKS = CNN.load_parameters()
        
        inp = input("\nSELECT COMMAND: T (TRAIN), TO (TEST ONE), TA (TEST ALL), D (DELETE), S (STOP), DS (DELETE AND STOP)\n")
        
        file_path = "/Users/arnavsarin/Desktop/CNN/"
    
     
        if(inp.strip().upper()== "T"):
        
            learning_rate = 0.1
     
            #ALL BATCHES
            for i in range (0,2450):
                for key in LEGO_BLOCKS:
                    cv_img = cv2.imread("/Users/arnavsarin/Desktop/CNN/25%_ROTATED_TRAINING/" + key +  str(LEGO_BLOCKS.get(key)).zfill(4) + ".png",0)
                    LEGO_BLOCKS[key] = LEGO_BLOCKS[key] + 1
                    actual = CNN.Feature_Learning(cv_img)
                    CNN.Classification()
                    expect = CNN.label(key)
                    new_parameters = CNN.Backpropogation(expect,learning_rate)
                    print(key + str(LEGO_BLOCKS[key] - 1))
             
            CNN.save_parameters(new_parameters,LEGO_BLOCKS)
    
    
        elif (inp.strip().upper() == "TA"):
    
            LEGO_BLOCKS_TEST_ALL = {"2x2-" : 1, "1x2-" : 1, "1x1-" : 1, "2x2plate-" : 1, "1x2plate-" : 1, "1x1plate-" : 1, "rooftile-" : 1, "peg2-" : 1, "lever-" : 1, "halfbush-" : 1 }
            
            count_correct = 0.0
            
            for i in range (0,350):
                for key in LEGO_BLOCKS_TEST_ALL:
                    cv_img2 = cv2.imread("/Users/arnavsarin/Desktop/CNN/25%_350/" + key + str(LEGO_BLOCKS_TEST_ALL.get(key)).zfill(4) + ".png",0)
                    feature = CNN.Feature_Learning(cv_img2)
                    CNN.Classification()
                    result = CNN.determine_label(feature)
                    print(str(key) + " " + str(i))
                    if(result == key):
                        count_correct = count_correct + 1.0
                    LEGO_BLOCKS_TEST_ALL[key] = LEGO_BLOCKS_TEST_ALL[key] + 1
                    
            percent1 = round((count_correct/3500),6)*100.0
            print("MODEL IS " + str(format(percent1,'.3f')) + "% ACCURATE FOR OLD DATA" )

            count_correct2 = 0.0
            
            for i in range (0,50):
                for key in LEGO_BLOCKS_TEST_ALL:
                    cv_img3 = cv2.imread("/Users/arnavsarin/Desktop/CNN/25%_50/" + key + str(LEGO_BLOCKS_TEST_ALL.get(key)).zfill(4) + ".png",0)
                    feature2 = CNN.Feature_Learning(cv_img3)
                    CNN.Classification()
                    result2 = CNN.determine_label(feature2)
                    print(str(key) + " " + str(i))
                    if(result2 == key):
                        count_correct2 = count_correct2 + 1.0
                    LEGO_BLOCKS_TEST_ALL[key] = LEGO_BLOCKS_TEST_ALL[key] + 1
                    
            percent2 = round((count_correct2/500),6)*100.0
            print("MODEL IS " + str(format(percent2,'.3f')) + "% ACCURATE FOR NEW DATA" )
            
            count_correct_all = count_correct + count_correct2
            percent3 = round((count_correct_all/4000.0),6)*100.0
            print("MODEL IS " + str(format(percent3,'.3f')) + "% ACCURATE FOR ALL DATA" )


        elif (inp.strip().upper()== "D"):
            print("DELETING NPY FILES")
            print(file_path)
            CNN.delete_parameters(file_path)
                
        elif (inp.strip().upper()== "S"):
            end = True
        
        elif (inp.strip().upper()== "DS"):
            print("DELETING NPY FILES")
            print(file_path)
            CNN.delete_parameters(file_path)
            end = True
                
                
                
if __name__ == "__main__":
    main()
#    test = np.array([[[9,1,1],[1,1,1],[1,1,3]],[[8,1,1],[1,1,1],[1,1,2]],[[7,1,1],[1,1,1],[1,1,3]]])
#    print(test)
#    print()
#    test = np.pad(test,((0,0),(1,1),(1,1)),constant_values=0)
#    test2 = np.rot90(test,2,(1,2))
#    print(test)
#    print()
#    print("TEST2")
#    print(test2)


 

 
