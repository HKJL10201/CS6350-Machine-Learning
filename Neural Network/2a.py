import numpy as np
import math

def sigmoid(x):
    return 1/(1+math.e**(-x))

class NN3:
    def __init__(self,input=3,layer1=3,layer2=3,output=1) -> None:
        self.layer1=np.ones(layer1)
        self.layer2=np.ones(layer2)
        W=[]
        W.append(np.zeros(input*(layer1-1)))
        W.append(np.zeros(layer1*(layer2-1)))
        W.append(np.zeros(layer2*output))
        self.W=W
    
    def output(self,X):
        for i in range(1,len(self.layer1)):
            s=0
            for j in range(len(X)):
                s+=X[j]*self.W[0][j+(i-1)*len(X)]
            self.layer1[i]=sigmoid(s)
        for i in range(1,len(self.layer2)):
            s=0
            for j in range(len(self.layer1)):
                s+=self.layer1[j]*self.W[1][j+(i-1)*len(self.layer1)]
            self.layer2[i]=sigmoid(s)
        y=0
        for i in range(len(self.layer2)):
            y+=self.layer2[i]*self.W[2][i]
        return y

    def derivative_w(self,x,y_):
        y=self.output(x)
        print('Output layer:')
        for i in range(len(self.W[2])):
            print('w'+str(i%len(self.layer2))+str(int(i/len(self.layer2))+1)+': ',end='')
            print((y-y_)*self.layer2[i%len(self.layer2)])
        print('Layer 2:')
        for i in range(len(self.W[1])):
            print('w'+str(i%len(self.layer1))+str(int(i/len(self.layer1))+1)+': ',end='')
            print((y-y_)*self.W[2][int(i/len(self.layer1))+1]*self.layer1[i%len(self.layer1)])
        print('Layer 1:')
        for i in range(len(self.W[0])):
            print('w'+str(i%len(x))+str(int(i/len(x))+1)+': ',end='')
            s=0
            for j in range(1,len(self.layer2)):
                s+=self.W[2][j]*self.W[1][(j-1)*len(self.layer1)+int(i/len(self.layer1))+1]
            print((y-y_)*s*x[i%len(x)])



nn=NN3()
nn.W=[[-1,-2,-3,1,2,3],
[-1,-2,-3,1,2,3],
[-1,2,-1.5]]
'''print(nn.layer1)
print(nn.layer2)
print(nn.W[0][0])
print(nn.output([1,1,1]))'''
nn.derivative_w([1,1,1],1)