from batch_gradient import GradientDescent
# import random
def main():
	# data = [(35,10),(20,7),(40,13),(50,15),(25,8),(100,33),(80,22),(75,18),(85,25),(55,18),(60,19),(10,3)]
	data = [(2.5,2.0),(1.0,1.7),(3.0,3.3),(4.0,2.5),(1.5,1.8),(8.0,4.3),(7.0,3.2),(6.5,2.8),(7.5,3.5),(4.5,2.8),(5.0,2.9),(1.0,1.3)]
	theta=[10,2]
	g = GradientDescent(data,theta)
	g.gradient(3,5000,0.08,0.005)


if __name__ == "__main__":
    main()


 ###
# import numpy as np
# import random

# # m denotes the number of examples here, not the number of features
# def gradientDescent(x, y, theta, alpha, m, numIterations):
#     xTrans = x.transpose()
#     for i in range(0, numIterations):
#         hypothesis = np.dot(x, theta)
#         loss = hypothesis - y
#         # avg cost per example (the 2 in 2*m doesn't really matter here.
#         # But to be consistent with the gradient, I include it)
#         cost = np.sum(loss ** 2) / (2 * m)
#         print("Iteration %d | Cost: %f" % (i, cost))
#         # avg gradient per example
#         gradient = np.dot(xTrans, loss) / m
#         # update
#         theta = theta - alpha * gradient
#     return theta


# def genData(numPoints, bias, variance):
#     x = np.zeros(shape=(numPoints, 2))
#     y = np.zeros(shape=numPoints)
#     # basically a straight line
#     for i in range(0, numPoints):
#         # bias feature
#         x[i][0] = 1
#         x[i][1] = i
#         # our target variable
#         y[i] = (i + bias) + random.uniform(0, 1) * variance
#     return x, y

# # gen 100 points with a bias of 25 and 10 variance as a bit of noise
# x, y = genData(100, 25, 10)
# m, n = np.shape(x)
# numIterations= 1000
# alpha = 0.0005
# theta = np.ones(n)
# theta = gradientDescent(x, y, theta, alpha, m, numIterations)
# print(theta)


###
# import numpy as np
# import random
# from sklearn.datasets.samples_generator import make_regression 
# import pylab
# from scipy import stats

# def gradient_descent_2(alpha, x, y, numIterations):
#     m = x.shape[0] # number of samples
#     theta = np.ones(2)
#     x_transpose = x.transpose()
#     for iter in range(0, numIterations):
#         hypothesis = np.dot(x, theta)
#         loss = hypothesis - y
#         J = np.sum(loss ** 2) / (2 * m)  # cost
#         print ("iter %s | J: %.3f" % (iter, J)    )  
#         gradient = np.dot(x_transpose, loss) / m         
#         theta = theta - alpha * gradient  # update
#     return theta

# if __name__ == '__main__':

# x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
#                         random_state=0, noise=35)
#     m, n = np.shape(x)
#     x = np.c_[ np.ones(m), x] # insert column
#     alpha = 0.01 # learning rate
#     theta = gradient_descent_2(alpha, x, y, 1000)

#     # plot
#     for i in range(x.shape[1]):
#         y_predict = theta[0] + theta[1]*x 
#     pylab.plot(x[:,1],y,'o')
#     pylab.plot(x,y_predict,'k-')
#     pylab.show()
#     print ("Done!")

