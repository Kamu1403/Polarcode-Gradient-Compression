import numpy as np
import matplotlib.pyplot as plt
from Compressor import PolarCompress, PolarCompressExtend, PolarCompressSplit, PolarCompressShorten, PolarCompressFinal
import torch
import math
import sympy as sp
from scipy import stats, optimize
from scipy.stats import ttest_1samp
from polar_compression import PolarCompressFinal
import time


# import tensorflow as tf


# def seperate():
#     from polarcodes import *
#     from BSC import AWGN_pure
#     from Encode import Compress
#     from Decode import Decompress
#
#     # initialise polar code
#     myPC = PolarCode(256, 100)
#     myPC.construction_type = 'bb'
#
#     # mothercode construction
#     design_SNR = 5.0
#     Construct(myPC, design_SNR)
#     print(myPC, "\n\n")
#
#     # set message
#     my_message = np.random.randint(2, size=myPC.M)
#     AWGN_pure(myPC, design_SNR, my_message)
#     print("The message is:", my_message)
#
#     # encode message
#     Compress(myPC)
#     print("The coded message is:", myPC.message_received)
#
#     # transmit the codeword
#     myPC.set_message(myPC.message_received)
#
#     # decode the received codeword
#     Decompress(myPC)
#     print("The decoded message is:", myPC.get_codeword())
#     print("Precision:", np.count_nonzero(my_message == myPC.get_codeword()) / my_message.shape[0])
#     print("Diffs:", np.where(my_message != myPC.get_codeword()))

def print_plt(array, name):
    plt.figure()
    plt.plot(array)
    plt.title(name)


def loglog_distribution(x: np.ndarray, a, b):
    pdf = (b / a * ((x / a) ** (b - 1))) / ((1 + (x / a) ** b) ** 2)
    cdf = (x ** b) / (a ** b + x ** b)
    return pdf, cdf


def lognormal_distribution(x: np.ndarray, u, o):
    pdf = 1 / (x * o * ((2 * np.pi) ** 0.5)) * np.exp(-(np.log(x) - u) ** 2 / (2 * (o ** 2)))
    cdf = 0.5 * (1 + np.vectorize(math.erf)((np.log(x) - u) / (o * (2 ** 0.5))))
    return pdf, cdf


def mle_loglog_distribution(x: np.ndarray, a, b):
    res = -(np.average(np.log(x) * (b - 1) - 2 * np.log(1 + (x / a) ** b)) + np.log(b / a) - (b - 1) * np.log(a))
    # print(res, a, b)
    if a <= 0 or b <= 0:
        return 100
    return res


def mle_lognormal_distribution(x: np.ndarray, u, o):
    res = -(np.average(-np.log(x) - 0.5 * ((np.log(x) - u) / o) ** 2) - np.log(o))
    # print(res, u, o)
    if o <= 0:
        return 100
    return res


def t_test_lognormal_distribution(x: np.ndarray, alpha, u, o):
    y = np.log(x)
    interval = stats.norm.interval(
        alpha=alpha,
        loc=np.mean(y),
        scale=stats.sem(y)
    )
    test_t, test_p = ttest_1samp(y, popmean=u)
    return test_p, interval


def kld_entropy(p, q):
    kld = p.copy()
    kld[p <= 0] = 0
    kld[p > 0] = p[p > 0] * np.log(p[p > 0] / q[p > 0])
    return np.sum(kld)


def ad_entropy(p, q, alpha=0.5):
    ad = p.copy()
    ad[p <= 0] = 0
    ad[p > 0] = (p[p > 0] ** alpha) / (q[p > 0] ** (alpha - 1))
    return np.log(np.sum(ad)) / (alpha - 1)


def tail_adaptive_kld_entropy(p, q):
    # todo:可能存在问题
    weight = get_tail_adaptive_weights(np.log(p[p > 0]), np.log(q[p > 0]))
    weight *= weight.shape[0]
    print_plt(weight, "tail_adaptive_weight")
    kld = p.copy()
    kld[p <= 0] = 0
    kld[p > 0] = p[p > 0] * np.log(p[p > 0] / q[p > 0]) * weight
    return np.sum(kld)


def get_tail_adaptive_weights(l_p, l_q, beta=-1.):
    """returns the tail-adaptive weights
    Args:
        l_p: log p(x), 1-d tensor, log probability of p
        l_q: log q(x), 1-d tensor, log probability of q
        beta: magnitude, default -1
    Returns:
        Tail-adaptive weights
    """
    diff = l_p - l_q
    diff -= np.max(diff)
    dx = np.exp(diff)
    prob = np.sign(np.expand_dims(dx, 1) - np.expand_dims(dx, 0))
    prob = np.greater(prob, 0.5).astype(np.float32)
    wx = np.sum(prob, axis=1) / np.size(l_p)
    wx = (1. - wx) ** beta  # beta = -1; or beta = -0.5

    wx /= np.sum(wx)  # self-normalization
    return wx


# def get_tail_adaptive_weights_tf(l_p, l_q, beta=-1.):
#     """returns the tail-adaptive weights
#     Args:
#         l_p: log p(x), 1-d tensor, log probability of p
#         l_q: log q(x), 1-d tensor, log probability of q
#         beta: magnitude, default -1
#     Returns:
#         Tail-adaptive weights
#     """
#     diff = l_p - l_q
#     diff -= tf.reduce_max(diff)
#     dx = tf.exp(diff)
#     prob = tf.sign(tf.expand_dims(dx, 1) - tf.expand_dims(dx, 0))
#     prob = tf.cast(tf.greater(prob, 0.5), tf.float32)
#     wx = tf.reduce_sum(prob, axis=1) / tf.cast(tf.size(l_p), tf.float32)
#     wx = (1. - wx) ** beta  # beta = -1; or beta = -0.5
#
#     wx /= tf.reduce_sum(wx)  # self-normalization
#     return tf.stop_gradient(wx)


def draw_pdf(dis_list):
    ran_list = np.asarray(dis_list) + 1
    count = np.bincount(ran_list) / ran_list.shape[0]
    cdf = np.cumsum(count)
    ran_range = np.arange(ran_list.max() + 1)

    res = optimize.minimize(lambda param, data: mle_loglog_distribution(data, param[0], param[1]),
                            x0=np.array([2000.0, 0.7]), args=(ran_list,), bounds=((0.1, 10000), (0.1, 10)),
                            method='BFGS').x
    print(res)
    loglog_pdf, loglog_cdf = loglog_distribution(ran_range, *res)
    print('loglog_kld=', kld_entropy(count, loglog_pdf))
    print('loglog_adaptive_kld=', tail_adaptive_kld_entropy(count, loglog_pdf))
    print('loglog_ad0.5=', ad_entropy(count, loglog_pdf, 0.5))
    print('loglog_adx2=', ad_entropy(count, loglog_pdf, 2))
    print('loglog_ad10=', ad_entropy(count, loglog_pdf, 10))

    res = optimize.minimize(lambda param, data: mle_lognormal_distribution(data, param[0], param[1]),
                            x0=np.array([7.2, 3.0]), args=(ran_list,), bounds=((0.1, 20), (0.1, 10)),
                            method='BFGS').x
    print(res)
    alpha = 0.95
    lognormal_pdf, lognormal_cdf = lognormal_distribution(ran_range, *res)
    lognormal_p, interval = t_test_lognormal_distribution(ran_list, alpha, *res)
    print('lognormal_kld=', kld_entropy(count, lognormal_pdf))
    print('lognormal_adaptive_kld=', tail_adaptive_kld_entropy(count, lognormal_pdf))
    print('lognormal_ad0.5=', ad_entropy(count, lognormal_pdf, 0.5))
    print('lognormal_adx2=', ad_entropy(count, lognormal_pdf, 2))
    print('lognormal_ad10=', ad_entropy(count, lognormal_pdf, 10))
    print('lognormal_p=', lognormal_p, ',reject' if lognormal_p < 0.05 else ',confirm')
    print('interval: ', interval)
    print(np.bincount(ran_list))

    plt.figure()
    plt.plot(ran_range, count, label="PDF")
    plt.plot(ran_range, cdf, label="CDF")
    plt.ylim(0, 1.1)
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("CDF for discrete distribution")
    plt.legend()

    plt.figure()
    plt.plot(ran_range, count, label="Noise")
    plt.plot(ran_range, loglog_pdf, label="loglog")
    plt.plot(ran_range, lognormal_pdf, label="lognormal")
    plt.ylim(0, 0.1)
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("PDF")
    plt.legend()

    plt.figure()
    plt.plot(ran_range, cdf, label="Noise")
    plt.plot(ran_range, loglog_cdf, label="loglog")
    plt.plot(ran_range, lognormal_cdf, label="lognormal")
    plt.ylim(0, 1.1)
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("CDF")
    plt.legend()

    plt.show()


def main():
    choose = 5

    # x = np.arange(0.25, 1, 0.01)
    # y = binary_entropy(x)
    # plt.plot(y,x)
    # plt.show()
    t_size = 4 * 4 * 4 * 3
    t_shape = (4, 4, 4, 3)

    if choose == 0:
        message = (torch.randn(t_size) * 64 * 64).type(torch.int16).reshape(*t_shape)
        # print(message)
        package = PolarCompressExtend.compress(message)
        print('package:', package)
        dec_message = PolarCompressExtend.decompress(*package)
        # print(dec_message)
        diff = message - dec_message
        draw_pdf(diff.type(dtype=torch.int64).abs().reshape(t_size))
        print('diff:', diff)
        print('avg diff:', diff.type(dtype=torch.int64).abs().type(dtype=torch.float).mean())
        print('rmse diff:',
              torch.nn.MSELoss()(message.type(dtype=torch.float), dec_message.type(dtype=torch.float)) ** 0.5)


    elif choose == 2:
        message = (torch.randn(t_size) * 64 * 64).type(torch.int16).reshape(*t_shape)
        # print(message)
        package = PolarCompressSplit.compress(message)
        print('package:', package)
        dec_message = PolarCompressSplit.decompress(*package)
        # print(dec_message)
        diff = message - dec_message
        draw_pdf(diff.type(dtype=torch.int64).abs().reshape(t_size))
        print('diff:', diff)
        print('avg diff:', diff.type(dtype=torch.int64).abs().type(dtype=torch.float).mean())
        print('rmse diff:',
              torch.nn.MSELoss()(message.type(dtype=torch.float), dec_message.type(dtype=torch.float)) ** 0.5)


    elif choose == 1:
        rate = [float(_) / 10 for _ in range(1, 10, 2)]
        n_range = [200, 500, 1000, 2000]
        n_colors = ['ro', 'go', 'bo', 'ko']
        store = np.ndarray([len(n_range), len(rate)])
        plt.axis([0, 1, 0, 0.5])
        plt.title('Rate-distortion')
        plt.xlabel('rate')
        plt.ylabel('distortion')
        for i, r in enumerate(rate):
            for j, n in enumerate(n_range):
                distortion = get_distortion(r, n, 3)
                plt.plot(r, distortion, n_colors[j])
                store[j, i] = distortion
                print('rate {}, n {} finished'.format(r, n))

        plt.show()
        print(store)


    elif choose == 4:
        compressor = PolarCompressShorten(200, 100, 5.0)
        my_message = np.random.randint(2, size=200)
        print("The message is:", my_message)

        message_received = compressor.compress(my_message)
        print("The coded message is:", message_received)

        message_end = compressor.decompress(message_received)
        print("The decoded message is:", message_end)
        print("Precision:", np.count_nonzero(my_message == message_end) / my_message.shape[0])
        print("Diffs:", np.where(my_message != message_end))


    elif choose == 5:
        for test_size in [2 ** 6, 2 ** 7, 2 ** 8, 2 ** 9, 2 ** 10, 2 ** 11]:
            message = (torch.randn(test_size)).type(torch.float32)
            # print(message)
            st_time = time.time()
            package = PolarCompressFinal.compress(message)
            dec_message = PolarCompressFinal.decompress(*package)
            # print(dec_message)
            ed_time = time.time()
            print("Time: ", ed_time - st_time, ", Size: ", test_size, ", Time/(N*log(N)): ",
                  (ed_time - st_time) / (test_size * math.log(test_size)))

    else:
        message = (torch.randn(t_size) * 64 * 64).type(torch.int32).reshape(*t_shape)
        # print(message)
        package = PolarCompressFinal.compress(message)
        print('package:', package)
        dec_message = PolarCompressFinal.decompress(*package)
        # print(dec_message)
        diff = message - dec_message
        # draw_pdf(diff.type(dtype=torch.int64).abs().reshape(t_size))
        print('diff:', diff)
        print('avg diff:', diff.type(dtype=torch.int64).abs().type(dtype=torch.float).mean())
        print('rmse diff:',
              torch.nn.MSELoss()(message.type(dtype=torch.float), dec_message.type(dtype=torch.float)) ** 0.5)


def get_distortion(rate, N, loops=10, SNR=5.0):
    K = int(N * rate)
    compressor = PolarCompressShorten(N, K, SNR)
    sums = np.zeros(1)
    for i in range(loops):
        my_message = np.random.randint(2, size=N)
        message_received = compressor.compress(my_message)
        message_end = compressor.decompress(message_received)
        sums += np.count_nonzero(my_message != message_end) / my_message.shape[0]
    return sums / loops


def binary_entropy(w):
    return 0.5 * np.log2(1 / w)


if __name__ == '__main__':
    main()
