import numpy as np
from time import time
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
import cv2


def euclidean_numpy(x, y):      # 单个数据 x 和 y 之间的距离
    return np.sqrt(np.sum(np.square(x-y)))


def neighbours(data, k):
    # 给定一个数据集 data，返回每个元素的 k 近邻对象，返回值为一个矩阵，行数为 data 的个数，列数为 k 近邻对象在 data 中的序号

    # 第一种方法，用时间换空间（占用空间小但耗时长）：
    distance_matrix = np.zeros((data.shape[0], data.shape[0]))  # 初始化每个对象间的距离矩阵
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            # distance_matrix[i, j] = euclidean(data[i, :], data[j, :])
            distance_matrix[i, j] = euclidean_numpy(data[i, :], data[j, :])
    distance_matrix += 1e10 * np.eye(data.shape[0])  # 给对角线的元素加的足够大，避免出现误排

    # # 第二种方法，用空间换时间（占用空间大，至少20.2G，办公室电脑16G不够用）：
    # raw_data = torch.from_numpy(data)
    # x = raw_data.expand(raw_data.shape[0], raw_data.shape[0], raw_data.shape[1])
    # y = raw_data.reshape([raw_data.shape[0], 1, raw_data.shape[1]]).repeat(1, raw_data.shape[0], 1)
    # distance_matrix = torch.sqrt((x - y).pow(2).sum(dim=2)).detach().numpy() + 1e10 * np.eye(raw_data.shape[0])

    sort_distance = np.sort(distance_matrix)                        # 对距离矩阵的每一行进行排序
    threshold = sort_distance[:, k-1]                               # 提取每一行的第 k 列的数值
    data_neighbours = np.zeros((data.shape[0], k))                  # 初始化最终要返回的对象矩阵
    for i in range(data.shape[0]):
        term = np.argwhere(distance_matrix[i, :] <= threshold[i]).ravel()
        data_neighbours[i, :] = term[0:k]
    return data_neighbours


def one_element_neighbour(test, data, k):       # 检索单一数据 test 在众多数据 data 中的 k 近邻，返回在 data 中的索引序号
    # 第一种方法，用时间（多次循环）换空间
    # distance_matrix = np.array([euclidean(test, data[i, :]) for i in range(data.shape[0])])

    # 第二种方法，用空间换时间
    test = test.reshape([1, len(test)]).repeat(data.shape[0], axis=0)
    distance_matrix = np.sqrt(((test - data) ** 2).sum(axis=1))

    sort_distance = np.sort(distance_matrix)    # 对距离矩阵的进行从小到大排序
    data_neighbours = np.argwhere(distance_matrix <= sort_distance[k-1]).ravel()
    return data_neighbours


def weighted_lever_correct_id(data, k, lambda_low, lambda_high, correct_id):
    # print("开集识别 lever 算法运行开始：", '\n')
    start_time = time()

    # 提取数据样本的 k 近邻
    k = k
    neighbour = neighbours(data, k)
    print("所有库内数据点的 k 近邻均已找到，所花时间为：{:.2f} sec".format(time() - start_time), '\n')

    # 计算 HBalance 系数
    h_balance = np.zeros(data.shape[0])     # 初始化 HBalance 系数矩阵
    for i in range(data.shape[0]):          # 对每个数据开始考虑
        term = data[neighbour[i, :].astype(int), :] - data[i, :]
        # h_balance[i] = np.square((term * np.exp(abs(term))).mean(0)).sum()
        h_balance[i] = np.square((term * abs(term)).mean(0)).sum()

    # 计算 divergence 系数
    divergence = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        divergence[i] = (euclidean_numpy(data[i, :], data[neighbour[i, :].astype(int), :])).sum()

    # 计算 DHBalance 系数
    d_h_balance = h_balance * divergence

    alpha = d_h_balance
    location_sorted = np.argsort(alpha)

    lambda_1 = lambda_low
    lambda_2 = lambda_high

    location = location_sorted[int((1-lambda_2)*len(alpha)):int((1-lambda_1)*len(alpha))]
    # print(location)

    # plt.plot(np.arange(0, len(alpha)), alpha)
    # plt.hist(alpha, bins=100)
    # plt.show()

    count = 0
    result = {}
    for i in range(len(location)):
        if (location[i]+1) in correct_id:
            count += 1
    result['correct points'] = count
    result['points detect'] = len(location)
    result['real boundary points'] = len(correct_id)
    result['precision'] = result['correct points'] / result['points detect']
    result['recall'] = result['correct points'] / result['real boundary points']
    if result['precision'] != 0 and result['recall'] != 0:
        result['F-measure'] = 2 / (1 / result['precision'] + 1 / result['recall'])
    return result


def weighted_lever_2d(data, k, lambda_low, lambda_high):
    # print("开集识别 lever 算法运行开始：", '\n')
    start_time = time()

    # 提取数据样本的 k 近邻
    k = k
    neighbour = neighbours(data, k)
    print("所有库内数据点的 k 近邻均已找到，所花时间为：{:.2f} sec".format(time() - start_time), '\n')

    # 计算 HBalance 系数
    h_balance = np.zeros(data.shape[0])     # 初始化 HBalance 系数矩阵
    for i in range(data.shape[0]):          # 对每个数据开始考虑
        term = data[neighbour[i, :].astype(int), :] - data[i, :]
        # h_balance[i] = np.square((term * np.exp(abs(term))).mean(0)).sum()
        h_balance[i] = np.square((term * abs(term)).mean(0)).sum()

    # 计算 divergence 系数
    divergence = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        divergence[i] = (euclidean_numpy(data[i, :], data[neighbour[i, :].astype(int), :])).sum()

    # 计算 DHBalance 系数
    d_h_balance = h_balance * divergence

    alpha = d_h_balance
    location_sorted = np.argsort(alpha)

    lambda_1 = lambda_low
    lambda_2 = lambda_high

    location = location_sorted[int((1-lambda_2)*len(alpha)):int((1-lambda_1)*len(alpha))]
    outlier = location_sorted[int((1-lambda_1)*len(alpha)):]
    return location, outlier


def colorful_transform(image):      # [H, W, C] image -> [H, W]
    result = np.zeros([image.shape[0], image.shape[1]])
    result[:, :] = 0.11*image[:, :, 0] + 0.59*image[:, :, 1] + 0.3 * image[:, :, 2]
    return result


if __name__ == '__main__':
    # ########################## 2D photo edge detection ###################################################

    # image1 = cv2.imread("./image/image1.jpeg")
    # print(image1.shape)
    # cv2.imshow("image1", image1)
    # cv2.waitKey()
    # trans_image1 = colorful_transform(image1)
    # cv2.imshow("trans_image1", trans_image1.astype("uint8"))
    # cv2.waitKey()
    # original_data = image1.reshape([-1, image1.shape[2]])
    # print(original_data.shape)

    # #################################
    from sklearn.datasets import make_blobs
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname='/System/Library/Fonts/Hiragino Sans GB.ttc')

    plt.figure(1)
    x1, y1 = make_blobs(n_samples=700, n_features=2, centers=3, cluster_std=1.5, random_state=10)
    plt.scatter(x1[np.argwhere(y1 == 0), 0], x1[np.argwhere(y1 == 0), 1], marker='o', label=u'聚类1')
    plt.scatter(x1[np.argwhere(y1 == 1), 0], x1[np.argwhere(y1 == 1), 1], marker='^', label=u'聚类2')
    plt.scatter(x1[np.argwhere(y1 == 2), 0], x1[np.argwhere(y1 == 2), 1], marker='s', label=u'聚类3')
    # plt.axis('off')
    plt.legend(prop=font)

    plt.figure(2)
    num_k = 3
    lambda1 = 0.02
    lambda2 = 0.17
    edge_location, outlier_location = weighted_lever_2d(x1, num_k, lambda1, lambda2)
    plt.scatter(x1[np.argwhere(y1 == 0), 0], x1[np.argwhere(y1 == 0), 1], marker='o', label=u'聚类1')
    plt.scatter(x1[np.argwhere(y1 == 1), 0], x1[np.argwhere(y1 == 1), 1], marker='^', label=u'聚类2')
    plt.scatter(x1[np.argwhere(y1 == 2), 0], x1[np.argwhere(y1 == 2), 1], marker='s', label=u'聚类3')
    plt.scatter(x1[edge_location, 0], x1[edge_location, 1], marker='X', edgecolors='black', label=u'边界点')
    plt.scatter(x1[outlier_location, 0], x1[outlier_location, 1], edgecolors='black', marker='d',
                label=u'噪声点', s=45)
    # plt.legend(['cluster1', 'cluster2', 'cluster3', 'edge data', 'outlier'], loc='best')
    plt.legend(prop=font)

    # # plt.axis('off')
    #
    plt.show()

    # ################################# biomed and cancer data processing ####################################
    # biomed_data = pd.read_table('./lever/biomed.txt', header=None)
    # biomed_correct_id = [13, 25, 46, 47, 51, 57, 55, 22, 72, 4, 6, 28, 35, 59, 48, 71, 11, 66, 197, 21,
    #                      32, 63, 12, 62, 23, 24, 65, 172, 64, 14]
    # cancer_data = pd.read_table('./lever/cancer.txt', header=None)
    # cancer_correct_id = cid = [72, 36, 40, 3, 33, 231, 143, 217, 16, 48, 122, 45, 166, 81, 158, 172, 119, 224,
    #                            35, 194, 105, 216, 23, 101, 95, 209, 162, 126, 193, 221, 191, 50, 56, 226, 180,
    #                            198, 131]
    #
    # data_biomed = biomed_data.values
    # data_cancer = cancer_data.values
    # print(data_biomed.shape, data_cancer.shape)
    # # data_cancer = data_cancer[0:367, :]
    #
    # num_k = 3
    # lambda1 = 0.003
    # lambda2 = 0.145
    # pprint(weighted_lever_correct_id(data_biomed, num_k, lambda1, lambda2, biomed_correct_id))
    #
    # num_k = 10
    # lambda1 = 0.4
    # lambda2 = 0.6
    # pprint(weighted_lever_correct_id(data_cancer, num_k, lambda1, lambda2, cancer_correct_id))
