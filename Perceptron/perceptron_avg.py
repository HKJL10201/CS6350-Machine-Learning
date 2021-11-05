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
    weight = [0, 0, 0, 0]
    bias = 0
    a = [0, 0, 0, 0, 0]

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
            a[0] += weight[0]
            a[1] += weight[1]
            a[2] += weight[2]
            a[3] += weight[3]
            a[4] += bias
        print('epoch %d, error rate: %f' % (i, err/summ))

    print("stop training: ")
    print(a)

    return a


def test(test_data, a):
    summ = 0
    err = 0
    random.shuffle(test_data)
    for data in test_data:
        summ += 1
        x1, x2, x3, x4, y = data
        predict = sign(a[0] * x1 + a[1] * x2 + a[2] * x3 + a[3] * x4 + a[4])
        if y * predict <= 0:
            err += 1
    print('test error rate: %f' % (err/summ))


def main():
    a = training(load_data(train_file))
    test(load_data(test_file), a)


if __name__ == '__main__':
    main()
