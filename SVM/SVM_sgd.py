import random
import csv
import numpy as np

T = 100
train_file = 'train.csv'
test_file = 'test.csv'
gamma_0 = 0.05
a=0.02
C=[100/873,500/873,700/873]

def sign(n):
    if n > 0:
        return 1
    else:
        return -1

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

def train(train_data,C):
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
            
            J=np.row_stack((W[:-1],[[0]]))
            if y*W.T*X<=1:
                err+=1
                J=J-C*N*y*X
            W=W-gamma_t(i+1)*J
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
    global train_file,test_file, C
    for c in C:
        print("When C = %f"%c)
        W=train(load_data(train_file),c)
        print('W = '+str(W.T))
        test(load_data(test_file),W)

if __name__ == '__main__':
    main()
