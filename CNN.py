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
        
        

    def __init__(self):
        
        #FILTER DIMENSIONS (KERNEL)
        self.kernel_height = 3
        self.kernel_length = 3
        
        #NUMBER OF FILTERS PER CONVOLUTION LAYER
        self.num_filtersL1 = 8
        self.num_filtersL2 = 12
        
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
        
        #CALCULATE THE VALUE OF CONVOLUTION LAYER 1
        self.C1_height = self.input_height - self.kernel_height + 1
        self.C1_length = self.input_length - self.kernel_length + 1
        self.C1 = np.zeros((self.num_filtersL1, self.C1_height, self.C1_length))
        
        #INITIALIZE SUBSET FOR FILTERED IMAGE
#        filtered_subset = np.zeros((self.kernel_height,self.kernel_length))
        
        #RUN THROUGH EACH FILTER
#        for f in range (self.num_filtersL1):
#            #RUN THROUGH THE LENGTH OF NEW CONVOLUTION LAYER
#            for l in range (self.C1_length):
#                #RUN THROUGH HEIGHT OF NEW CONVOLUTION LAYER
#                for h in range (self.C1_height):
#                    #MULTIPYING FILTER TO SLICED IMAGE
#                    filtered_subset = input_img[h:h+self.kernel_height,l:l+self.kernel_length] * self.filterL1[f]
#
#                    #SUM OF FILTERED IMAGE SUBSET
#                    self.C1[f,h,l] = np.sum(filtered_subset)
                    
        #CONVOLUTION FUNCTION PARALLELISM
        self.C1 = self.conv_parallelism(self.filterL1,input_img,self.C1)


        #ReLU ACTIVATION FUNCTION
        self.activ_C1 = self.ReLU(self.C1)

        #CALCULATE THE VALUE OF POOLING LAYER 1
        self.P1_height = int(self.C1_height/ self.pool_height)
        self.P1_length = int(self.C1_length/ self.pool_length)
        self.P1 = np.zeros((self.num_filtersL1, self.P1_height, self.P1_length))
        
        #RUN THROUGH EACH CONVOLUTION FOR LAYER 1
        for f in range (self.num_filtersL1):
            #RUN THROUGH THE LENGTH OF NEW POOLING LAYER
            for l in range (self.P1_length):
                #RUN THROUGH HEIGHT OF NEW POOLING LAYER
                for h in range (self.P1_height):
                    #MAX VALUE OF POOL CALCULATION
                    h_iter = h * self.pool_height
                    l_iter = l * self.pool_length
                    self.P1[f,h,l] = self.max_pooling(self.activ_C1[f, h_iter:h_iter+self.pool_height,l_iter:l_iter+self.pool_length])
                    
        
        #CALCULATE THE VALUE OF CONVOLUTION LAYER 2
        self.C2_height = int(self.P1_height - self.kernel_height + 1)
        self.C2_length = int(self.P1_length - self.kernel_length + 1)
        self.C2 = np.zeros((self.num_filtersL1 * self.num_filtersL2, self.C2_height, self.C2_length))
        
        #INITIALIZE SUBSET FOR FILTERED IMAGE
#        filtered_subset = np.zeros((self.kernel_height,self.kernel_length))
        
        #RUN THROUGH EACH FILTER
#        for f in range (self.num_filtersL1 * self.num_filtersL2):
#            #RUN THROUGH THE LENGTH OF NEW CONVOLUTION LAYER
#            for l in range (self.C2_length):
#                #RUN THROUGH HEIGHT OF NEW CONVOLUTION LAYER
#                for h in range (self.C2_height):
#                    #MULTIPYING FILTER TO SLICED IMAGE
#                    filtered_subset = self.P1[int(f/self.num_filtersL2),h:h+self.kernel_height,l:l+self.kernel_length] * self.filterL2[int(f/self.num_filtersL1)]
#
#                    #SUM OF FILTERED IMAGE SUBSET
#                    self.C2[f,h,l] = np.sum(filtered_subset)
                    
        
        #CONVOLUTION FUNCTION PARALLELISM
        self.C2 = self.conv_parallelism(self.filterL2,self.P1,self.C2)


        #ReLU ACTIVATION FUNCTION
        self.activ_C2 = self.ReLU(self.C2)
            
        #CALCULATE THE VALUE OF POOLING LAYER 2
        self.P2_height = int(self.C2_height/self.pool_height)
        self.P2_length = int(self.C2_length/self.pool_length)
        self.P2 = np.zeros((self.num_filtersL1 * self.num_filtersL2, self.P2_height, self.P2_length))
        
        #RUN THROUGH EACH CONVOLUTION FOR LAYER 2
        for f in range (self.num_filtersL1 * self.num_filtersL2):
            #RUN THROUGH THE LENGTH OF NEW POOLING LAYER
            for l in range (self.P2_length):
                #RUN THROUGH HEIGHT OF NEW POOLING LAYER
                for h in range (self.P2_height):
                    #MAX VALUE OF POOL CALCULATION
                    h_iter = h * self.pool_height
                    l_iter = l * self.pool_length
                    self.P2[f,h,l] = self.max_pooling(self.activ_C2[f, h_iter:h_iter+self.pool_height,l_iter:l_iter+self.pool_length])

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
        
        #CROSS ENTROPY LOSS
        error = -1 * (cost_vec/self.o1)
        
        #OUTPUT - HIDDEN LAYER 1
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
        
        #SUMMING THE LOSS TO HAVE SAME DIMENSIONS AS 1ST SET OF FILTERS
        loss2_out = np.split(loss2_out,self.num_filtersL1)
        loss2_out = np.sum(loss2_out,axis=0)
        
        
        #FILTER 2 CHANGE INITIALIZATION
        self.c_filter2 = np.zeros((self.num_filtersL2,self.kernel_height,self.kernel_length))
        
        #NUMBER OF FILTERS ITERATION
#        for f in range (self.num_filtersL2 * self.num_filtersL1):
#            #LENGTH OF LOSS DERIVATIVE
#            for l in range (self.kernel_length):
#                #HEIGHT OF LOSS DERIVATIVE
#                for h in range (self.kernel_height):
#                    #LOSS HEIGHT AND LOSS LENGTH
#                    h_iter = self.P1_height - self.kernel_height + 1
#                    l_iter = self.P1_length - self.kernel_length + 1
#
#                    #MULTIPLYING LOSS TO SLICED IMAGE
#                    pool_subset2 = self.P1[int(f/self.num_filtersL2),h:h+h_iter,l:l+l_iter] * loss2_out[int(f/self.num_filtersL1),h,l]
#
#                    #SUM OF MULTIPLIED LOSS SUBSET
#                    self.c_filter2[int(f/self.num_filtersL1),h,l] = np.sum(pool_subset2)
        

        #CONVOLUTION FUNCTION PARALLELISM
        self.c_filter2 = self.conv_parallelism(loss2_out,self.P1,self.c_filter2)

        
        #NEED TO PAD LOSS
        padded_loss2 = np.pad(loss2_out,self.kernel_length-1,mode='constant')
        
        #ERROR BEFORE CONVOLUTION 2
        C2_error = np.zeros((self.num_filtersL1,self.P1_height,self.P1_length))
        
        #ITERATION THROUGH FILTERS
#        for f in range (self.num_filtersL1 * self.num_filtersL2):
#            #LENGTH OF FILTER
#            for l in range (self.P1_length):
#                #HEIGHT OF FILTER
#                for h in range (self.P1_height):
#                    #MULTIPYING FILTER TO SLICED IMAGE LOSS
#                    loss_subset2 = padded_loss2[int(f/self.num_filtersL1),h:h+self.kernel_height,l:l+self.kernel_length] * self.filterL2[int(f/self.num_filtersL1)]
#
#                    #THE C2_error MAY HAVE AN ERROR IN f/self.num_filtersL2------------------------------------------------------
#                    #SUM OF FILTERED IMAGE SUBSET
#                    C2_error[int(f/self.num_filtersL2),h,l] = np.sum(loss_subset2)
                    
        
        #CONVOLUTION FUNCTION PARALLELISM
        C2_error = self.conv_parallelism(self.filterL2,padded_loss2,C2_error)

                     
        #DERIVATIVE OF POOLING LAYER 2 AND ERROR MULTIPLICATION
        dpooling = self.dmax_pooling(self.activ_C1,self.P1,C2_error)
        
        #DERIVATIVE OF RELU
        dRelU = self.d_ReLU(self.activ_C1)
        
        #DERIVATIVE LOSS TO OUTPUT
        loss_out = dpooling * dRelU
        
        
        #FILTER 1 CHANGE INITIALIZATION
        self.c_filter = np.zeros((self.num_filtersL1,self.kernel_height,self.kernel_length))
        
        #NUMBER OF FILTERS ITERATION
#        for f in range (self.num_filtersL1):
#            #LENGTH OF LOSS DERIVATIVE
#            for l in range (self.kernel_length):
#                #HEIGHT OF LOSS DERIVATIVE
#                for h in range (self.kernel_height):
#                    h_iter = self.input_height - self.kernel_height + 1
#                    l_iter = self.input_length - self.kernel_length + 1
#
#                    #MULTIPLYING LOSS TO SLICED IMAGE
#                    pool_subset = self.P1[f,h:h + h_iter,l:l+ l_iter] * loss_out[f,h,l]
#
#                    #SUM OF MULTIPLIED LOSS SUBSET
#                    self.c_filter[f,h,l] = np.sum(pool_subset)


        #CONVOLUTION FUNCTION PARALLELISM
        self.c_filter = self.conv_parallelism(loss_out,self.P1,self.c_filter)
        
                    
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
            os.rmdir(file_path)
        
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


 

 
