import numpy as np
import random
import math
import csv
import warnings
 
warnings.filterwarnings('ignore')

T = 10
train_file = 'train.csv'
test_file = 'test.csv'
gamma_0 = 0.05
d=0.02

width=[5,10,25,50,100]

def gamma_t(t):
    global gamma_0,d
    return gamma_0/(1+(gamma_0/d)*t)

def sigmoid(x):
    return 1/(1+math.e**(-x))

class NN3:
    def __init__(self,input=3,layer1=3,layer2=3,output=1) -> None:
        self.layer1=np.ones(layer1)
        self.layer2=np.ones(layer2)
        W=[]
        W.append(np.random.normal(loc=0,scale=1.0,size=input*(layer1-1)))
        W.append(np.random.normal(loc=0,scale=1.0,size=layer1*(layer2-1)))
        W.append(np.random.normal(loc=0,scale=1.0,size=layer2*output))
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
    
    def update_w(self,x,y_,r):
        y=self.output(x)
        # Output layer
        for i in range(len(self.W[2])):
            self.W[2][i]-=r*(y-y_)*self.layer2[i%len(self.layer2)]
        # Layer 2
        for i in range(len(self.W[1])):
            self.W[1][i]-=r*(y-y_)*self.W[2][int(i/len(self.layer1))+1]*self.layer1[i%len(self.layer1)]
        # Layer 1
        for i in range(len(self.W[0])):
            s=0
            for j in range(1,len(self.layer2)):
                s+=self.W[2][j]*self.W[1][(j-1)*len(self.layer1)+int(i/len(self.layer1))+1]
            self.W[0]-=r*(y-y_)*s*x[i%len(x)]

def load_data(filename):
    f = open(filename, 'r')
    r = csv.reader(f)
    data = []
    for line in r:
        new_line = [float(i) for i in line[:-1]]
        if line[-1] == '0':
            new_line.append(-1)
        else:
            new_line.append(1)
        data.append(new_line)
    f.close()
    return data

def sign(n):
    if n > 0:
        return 1
    else:
        return -1

def train(train_data,network):
    for i in range(T):
        summ = 0
        err = 0
        random.shuffle(train_data)
        for data in train_data:
            summ += 1
            x1, x2, x3, x4, y_ = data
            X=[1,x1,x2,x3,x4]
            
            if sign(network.output(X))!=y_:
                err+=1
            network.update_w(X,y_,gamma_t(i+1))
        #print('epoch %d, error rate: %f' % (i, err/summ))
        if i==T-1:
            print('training error rate: %f' % (err/summ))

def test(test_data, network):
    summ = 0
    err = 0
    random.shuffle(test_data)
    for data in test_data:
        summ += 1
        x1, x2, x3, x4, y_ = data
        X=[1,x1,x2,x3,x4]

        if sign(network.output(X))!=y_:
                err+=1
    print('test error rate: %f' % (err/summ))

def main():
    global train_file,test_file,width
    for w in width:
        print("When width = %d"%w)
        nn=NN3(5,w,w)
        #print(nn.W)
        train(load_data(train_file),nn)
        #print(nn.W)
        test(load_data(test_file),nn)

if __name__ == '__main__':
    main()
