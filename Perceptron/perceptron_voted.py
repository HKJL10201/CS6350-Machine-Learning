import random
import csv

T = 10
train_file = 'train.csv'
test_file = 'test.csv'
learning_rate = 0.1


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


def training(train_data):
    global T, learning_rate
    W = []
    B = []
    C = []
    weight = [0, 0, 0, 0]
    bias = 0
    m = -1
    c = 0

    for i in range(T):
        summ = 0
        err = 0
        random.shuffle(train_data)
        for data in train_data:
            summ += 1
            x1, x2, x3, x4, y = data
            predict = sign(weight[0] * x1 + weight[1] * x2 +
                           weight[2] * x3 + weight[3] * x4 + bias)
            if y * predict <= 0:
                err += 1
                weight[0] = weight[0] + learning_rate * y * x1
                weight[1] = weight[1] + learning_rate * y * x2
                weight[2] = weight[2] + learning_rate * y * x3
                weight[3] = weight[3] + learning_rate * y * x4
                bias = bias + learning_rate * y
                new_w = []
                new_w.append(weight[0])
                new_w.append(weight[1])
                new_w.append(weight[2])
                new_w.append(weight[3])
                W.append(new_w)
                B.append(bias)
                C.append(c)
                c = 1
                m += 1
            else:
                c += 1
        print('epoch %d, error rate: %f' % (i, err/summ))

    print("stop training: ")
    for i in range(len(W)):
        print(W[i]+[B[i]], C[i])
    print(m+1)

    return W, B, C


def test(test_data, W, B, C):
    summ = 0
    err = 0
    random.shuffle(test_data)
    for data in test_data:
        summ += 1
        x1, x2, x3, x4, y = data
        for i in range(len(W)):
            weight = W[i]
            bias = B[i]
            c = C[i]
            res = 0
            res += c*sign(weight[0] * x1 + weight[1] * x2 +
                          weight[2] * x3 + weight[3] * x4 + bias)
        predict = sign(res)
        if y * predict <= 0:
            err += 1
    print('test error rate: %f' % (err/summ))


def main():
    W, B, C = training(load_data(train_file))
    test(load_data(test_file), W, B, C)


if __name__ == '__main__':
    main()
