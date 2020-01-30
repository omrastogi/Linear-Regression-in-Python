import numpy as np
import matplotlib.pyplot as plt

def grad_descend(x,y,theta0=0,theta1=0,alpha=0.0001,iter=100):
	m = len(x)
	for i in range(0,iter):
		h = hypothesis(theta0, theta1, x)
		j = cost_function(h,y)
		theta0 = theta0 - alpha*(np.sum(h-y))/m
		theta1 = theta1 - alpha*((h-y).dot(x))/m
		#print (theta0, theta1)
		#print (h)
		#print (j)
	return theta0,theta1

	
def hypothesis(theta0,theta1,x):
	#this is the fuction responsible to decide the behaviour of the model
	#hypotheis differs for different algorthitms/approches
	h = theta0+theta1*x
	return h

def cost_function(h,y):
	j = (1/(2*len(y)))*np.sum((h-y))
	return j 

