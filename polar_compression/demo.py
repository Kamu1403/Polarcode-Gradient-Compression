import math

import numpy as np
import matplotlib.pyplot as plt
from Compressor import PolarCompress, PolarCompressExtend, PolarCompressSplit
import torch


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
def loglog_distribution(x: np.ndarray, a, b):
    pdf = (b / a * ((x / a) ** (b - 1))) / ((1 + (x / a) ** b) ** 2)
    cdf = (x ** b) / (a ** b + x ** b)
    return pdf, cdf


def draw_pdf(dis_list):
    ran_list = np.asarray(dis_list)
    count = np.bincount(ran_list) / ran_list.shape[0]
    cdf = np.cumsum(count)
    ran_range = np.arange(ran_list.max() + 1)
    loglog_pdf, loglog_cdf = loglog_distribution(ran_range, 300, 0.7)
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
    plt.ylim(0, 0.1)
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("PDF")
    plt.legend()

    plt.figure()
    plt.plot(ran_range, cdf, label="Noise")
    plt.plot(ran_range, loglog_cdf, label="loglog")
    plt.ylim(0, 1.1)
    plt.xlabel("X")
    plt.ylabel("Probability Values")
    plt.title("CDF")
    plt.legend()

    plt.show()


def main():
    choose = 2

    # x = np.arange(0.25, 1, 0.01)
    # y = binary_entropy(x)
    # plt.plot(y,x)
    # plt.show()

    if choose == 0:
        message = (torch.randn(4 * 4 * 4 * 4) * 64 * 64).type(torch.int16).reshape(4, 4, 4, 4)
        # print(message)
        package = PolarCompressExtend.compress(message)
        print('package:', package)
        dec_message = PolarCompressExtend.decompress(*package)
        # print(dec_message)
        diff = message - dec_message
        draw_pdf(diff.type(dtype=torch.int64).abs().reshape(4 * 4 * 4 * 4))
        print('diff:', diff)
        print('avg diff:', diff.type(dtype=torch.int64).abs().type(dtype=torch.float).mean())
        print('rmse diff:',
              torch.nn.MSELoss()(message.type(dtype=torch.float), dec_message.type(dtype=torch.float)) ** 0.5)


    elif choose == 2:
        message = (torch.randn(4 * 4 * 4 * 4) * 64 * 64).type(torch.int16).reshape(4, 4, 4, 4)
        # print(message)
        package = PolarCompressSplit.compress(message)
        print('package:', package)
        dec_message = PolarCompressSplit.decompress(*package)
        # print(dec_message)
        diff = message - dec_message
        draw_pdf(diff.type(dtype=torch.int64).abs().reshape(4 * 4 * 4 * 4))
        print('diff:', diff)
        print('avg diff:', diff.type(dtype=torch.int64).abs().type(dtype=torch.float).mean())
        print('rmse diff:',
              torch.nn.MSELoss()(message.type(dtype=torch.float), dec_message.type(dtype=torch.float)) ** 0.5)


    elif choose == 1:
        rate = [float(_) / 10 for _ in range(1, 10, 2)]
        n_range = [256, 512, 1024, 2048]
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
    else:
        compressor = PolarCompress(256, 100, 5.0)
        my_message = np.random.randint(2, size=256)
        print("The message is:", my_message)

        message_received = compressor.compress(my_message)
        print("The coded message is:", message_received)

        message_end = compressor.decompress(message_received)
        print("The decoded message is:", message_end)
        print("Precision:", np.count_nonzero(my_message == message_end) / my_message.shape[0])
        print("Diffs:", np.where(my_message != message_end))


def get_distortion(rate, N, loops=10, SNR=5.0):
    K = int(N * rate)
    compressor = PolarCompress(N, K, SNR)
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
