import math
import numpy as np
'''
def sigmoid(x):
    return 1/(1+math.e**(0-x))

x=[1,1,1]
w11=[-1,-2,-3]
w12=[1,2,3]
w21=[-1,-2,-3]
w22=[1,2,3]
w3=[-1,2,-1.5]'''

def dJ(w,x,y):
    return math.exp(-y*w.T*x)/(1+math.exp(-y*w.T*x))*y*x+w

learning_rate=[0.01,0.005,0.0025]
X=[[0.5,-1,0.3],
    [-1,-2,-2],
    [1.5,0.2,-2.5]]
Y=[1,-1,1]
N=len(Y)
C=1/3

W=np.matrix([0,0,0,0]).T
for i in range(N):
    x=X[i]
    x.append(1)
    x=np.matrix(x).T
    y=Y[i]
    '''J=np.row_stack((W[:-1],[[0]]))
    if y*W.T*x<=1:
        J=J-C*N*y*x'''
    J=N*dJ(W,x,y)
    W=W-learning_rate[i]*J
    print(J.T)
    print(W.T)

'''
if __name__ == '__main__':
    #print(sigmoid(float(input())))
    a=-3.437*(2*(-3)+(-1.5)*3)*1
    print(a)
    pass
'''