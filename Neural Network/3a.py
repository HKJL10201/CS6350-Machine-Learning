import numpy as np
import random
import math
import csv


T = 100
train_file = 'train.csv'
test_file = 'test.csv'

gamma_0 = 0.05
a=0.02
V=[0.01,0.1,0.5,1,3,5,10,100]

def gamma_t(t):
    global gamma_0,a
    return gamma_0/(1+(gamma_0/a)*t)

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


def train(train_data,v):
    W=np.matrix([0,0,0,0,0]).T
    N=len(train_data)

    for i in range(T):
        summ = 0
        err = 0
        random.shuffle(train_data)
        for data in train_data:
            summ += 1
            x1, x2, x3, x4, y = data
            X=np.matrix([x1,x2,x3,x4,1]).T
            
            yy=float(y*W.T*X)
            #print(float(yy))
            ee=math.e**(-yy)
            dd=ee/(1+ee)*y*X+(2/v)*W
            if y*W.T*X<=1:
                err+=1
            W=W-gamma_t(i+1)*N*dd
        #print('epoch %d, error rate: %f' % (i, err/summ))
        if i==T-1:
            print('training error rate: %f' % (err/summ))
    return W

def test(test_data, W):
    summ = 0
    err = 0
    random.shuffle(test_data)
    for data in test_data:
        summ += 1
        x1, x2, x3, x4, y = data
        X=np.matrix([x1,x2,x3,x4,1]).T

        predict=sign(W.T*X)
        if y * predict <= 0:
            err += 1
    print('test error rate: %f' % (err/summ))

def main():
    global train_file,test_file, V
    for v in V:
        print("When V = %f"%v)
        W=train(load_data(train_file),v)
        print('W = '+str(W.T))
        test(load_data(test_file),W)

if __name__ == '__main__':
    main()
