# my own mini-batch gradient descent 

import random
import math
import numpy as np
import matplotlib.pyplot as plt
# import stastics

class GradientDescent(object):

    def __init__(self,data,theta):
        # theta = [theta0,theta1]  h(x)= theta0 + theta1*x
        # data -> [(x1,y1),(x2,y2),...]
        x,y = zip(*data)
        #研究一下正規化
        self.data = data
        self.x = x
        self.y = y
        self.theta = theta 

    def in_random_order(self):
        '''generator that returns the elements of data in random set'''
        indexes = [i for i,_ in enumerate(self.data)]
        random.shuffle(self.data)
        # for i in indexes:
            # yield self.data[i]  #你可以在函式中包括yield來「產生」值，表面上看來，yield就像是return會傳回值，但又不中斷函式的執行 但是用起來怪怪的??
        return self.data

    def gradient(self,mini_batch_size,iter_no,tol,alpha):
        ''' update theta until iter > iter_no or error < tol'''
        cost = 1 #that is J(theta) cost function
        iterno = 0
        data_size = len(self.data)
        while iterno < iter_no and cost > tol:
            cost = 0
            theta_origin = self.theta
            self.in_random_order()
            mini_batches = [self.data[k:k+mini_batch_size] for k in range(0,data_size,mini_batch_size)]
            # print (mini_batches)# batch_size個data一組的 2d array
            for mini_batch in mini_batches: 
                # print (mini_batch)  # batch_size個data一組的 2d array           
                self.update(mini_batch,alpha) # return update theta
            for xi,yi in self.data:
                hypothesis = self.theta[0] + xi*self.theta[1]
                cost += (hypothesis - yi)**2
            cost = cost/(2*data_size)

            print ("Iteration:",iterno," Error: ",float('%.5f'%(cost)))
            iterno +=1

    def update(self,mini_batch,alpha):
        diff0 = 0
        diff1 = 0
        cost = 0
        m = len(mini_batch)             
        for xi,yi in mini_batch:
            # print(m) # 3
            hypothesis = self.theta[0] + xi*self.theta[1]
            # print (hypothesis)
            diff0 += (hypothesis-yi)
            diff1 += (hypothesis-yi)*xi

        diff0= float('%.5f'%(diff0/m))
        diff1= float('%.5f'%(diff1/m))
        # print (cost)

        theta_iter0 = self.theta[0] - alpha*diff0
        theta_iter1 = self.theta[1] - alpha*diff1
        self.theta = [float('%.5f'%(theta_iter0)),float('%.5f'%(theta_iter1))]
        print (self.theta)