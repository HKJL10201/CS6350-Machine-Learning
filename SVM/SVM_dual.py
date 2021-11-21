import random
import csv
import numpy as np
from scipy.optimize import minimize

train_file = 'train.csv'
test_file = 'test.csv'
C=[100/873,500/873,700/873]

def func(X,Y):
    def f(a):
        fir=0
        sec=0
        for i in range(len(X)):
            for j in range(len(X)):
                fir+=Y[i]*Y[j]*a[i]*a[j]*X[i].T*X[j]
            sec+=a[i]
        return 0.5*fir-sec
    return f

def con(C,Y):
    cons=[]
    for i in range(len(Y)):
        cons.append({'type': 'ineq', 'fun': lambda a: a[i]})
        cons.append({'type': 'ineq', 'fun': lambda a: C-a[i]})
    def c(a):
        res=0
        for i in range(len(a)):
            res+=a[i]*Y[i]
        return res
    cons.append({'type': 'eq', 'fun': c})
    return cons

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

def matrixize(file):
    X=[]
    Y=[]
    train_data=load_data(file)
    for data in train_data:
        x1, x2, x3, x4, y = data
        x=np.matrix([x1,x2,x3,x4,1]).T
        X.append(x)
        Y.append(y)
    return X,Y

def main():
    global train_file,C
    X,Y=matrixize(train_file)
    cons=con(C[0],Y)
    a0=np.asarray([0 for i in range(len(Y))])
    res=minimize(func(X,Y),a0,method='SLSQP',constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)

if __name__ == '__main__':
    main()
