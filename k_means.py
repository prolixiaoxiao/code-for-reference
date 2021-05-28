import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def distance(vec1, vec2):
    return np.sqrt(np.sum(np.square(vec1 - vec2)))

def kmeans(data, k):
    K = np.random.uniform(0, 10, (k, data.shape[1]))
    ret = np.zeros((data.shape[0], 2))
    flag = True
    while flag:
        flag = False
        for i in range(data.shape[0]):
            # print("数据的每一行：{}".format(data[i]))
            minDist = np.inf
            minIndex = -1
            # print(K.shape[1])
            for j in range(K.shape[0]):
                # print("第{}个质心点{}".format(j, K[j]))
                # 计算数据中的每个点到聚类中心的距离
                ds = distance(data[i], K[j])
                # print("距离：{}".format(ds))
                if ds < minDist:
                    minDist = ds
                    minIndex = j
                    print("距离和簇中心:{}   {}".format(ds, str(j)))
            # 每次计算完一行数据到质心的距离后，更新ret矩阵的结果（将数据点分给距离其最近的簇）
            ret[i][0] = minDist
            ret[i][1] = minIndex
        # 对每个簇，计算簇中所有点的均值并将均值作为质心
        for i in range(k):
            cluster = data[ret[:, 1] == i]
            # print(cluster)
            if len(cluster) == 0:
                pass
            else:
                center = np.mean(cluster, axis=0)
                # print(center)
                if (center == K[i]).all():
                    pass
                else:
                    flag = True
                    K[i] = center
    # 质心不发生改变
    for i in range(len(K)):
        print("质心点为：{}".format(K[i]))
    data_c = np.c_[data, ret]
    # print(data_c)
    data_c = pd.DataFrame(data_c)
    data_c.to_csv('./ret.csv')

    center_x = K[:, 0]
    center_y = K[:, 1]
    plt.scatter(center_x, center_y, marker="X")
    plt.scatter(X, Y)
    plt.show()


if __name__ == '__main__':
    data1 = np.random.uniform(0, 2, (10, 2))
    data2 = np.random.uniform(3, 6, (10, 2))
    data3 = np.random.uniform(8, 10, (10, 2))
    data = np.r_[data1, data2, data3]
    np.random.shuffle(data)

    X = data[:, 0]
    Y = data[:, 1]
    plt.scatter(X, Y)
    plt.show()
    kmeans(data, 3)

