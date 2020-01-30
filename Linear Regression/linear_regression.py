import numpy as np
from gradientdescend import grad_descend
import matplotlib.pyplot as plt

theta0 = 0
theta1 = 1
h=[]
def main():
	train()
	test()

def train():
	points = np.genfromtxt("train.csv", delimiter=",")
	x = np.array(points[1:,0])
	y = np.array(points[1:,1]) 
	theta0 = 0
	theta1 = 0
	learning_rate = 0.0001
	iterations = 100
	theta0,theta1 = grad_descend(x,y,theta0,theta1,learning_rate,iterations)
	print (theta0,theta1)
	h = theta0+theta1*x
	#plt.plot (x,y,'ro')
	#plt.plot (x,h)
	#plt.title("Train")
	#plt.show()


def test():
	points = np.genfromtxt("test.csv", delimiter=",")
	x = np.array(points[1:,0])
	y = np.array(points[1:,1]) 
	h = theta0+theta1*x
	plt.plot(x,y,'ro')
	plt.plot(x,h)
	plt.title("Test")
	plt.show()


main()