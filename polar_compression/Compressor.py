import numpy
import numpy as np
import torch
from polarcodes import *
from BSC import AWGN_pure
from Encode import Compress
from Decode import Decompress
from komm._util import pack, unpack


class PolarCompress:
    def __init__(self, N, K, SNR=5.0):
        self.N = N
        self.K = K
        self.SNR = SNR

        # initialise polar code
        self.myPC = PolarCode(self.N, self.K)
        self.myPC.construction_type = 'bb'

        # mothercode construction
        Construct(self.myPC, SNR)
        # print(myPC, "\n\n")

    def compress(self, message):
        # set message
        AWGN_pure(self.myPC, self.SNR, message)
        # print("The message is:", message)

        # encode message
        Compress(self.myPC)
        # print("The coded message is:", myPC.message_received)

        return self.myPC.message_received

    def decompress(self, message_received):
        # transmit the codeword
        self.myPC.set_message(message_received)

        # decode the received codeword
        Decompress(self.myPC)
        # print("The decoded message is:", myPC.get_codeword())
        # print("Precision:", numpy.count_nonzero(my_message == myPC.get_codeword()) / my_message.shape[0])
        # print("Diffs:", numpy.where(my_message != myPC.get_codeword()))

        return self.myPC.get_codeword()


class PolarCompressExtend:
    compress_rate = 0.5

    @staticmethod
    def unit_size(dtype):
        if dtype == torch.int8:
            unit_size = 8
        elif dtype == torch.int16:
            unit_size = 16
        else:
            raise NotImplementedError('dtype={}'.format(dtype))
        return unit_size

    @staticmethod
    def compress(message: torch.Tensor):
        unit_size = PolarCompressExtend.unit_size(message.dtype)
        polar_n = unit_size * message.shape.numel()
        polar_k = int(polar_n * PolarCompressExtend.compress_rate)
        bit_message = unpack(message.reshape(message.shape.numel()), unit_size)

        compressor = PolarCompress(polar_n, polar_k)
        return pack(compressor.compress(bit_message), 32), (message.shape, message.dtype)

    @staticmethod
    def decompress(bit_message, ctx):
        shape, dtype = ctx
        unit_size = PolarCompressExtend.unit_size(dtype)
        polar_n = unit_size * shape.numel()
        bit_message = unpack(bit_message, 32)
        compressor = PolarCompress(polar_n, bit_message.shape[0])
        bit_message = compressor.decompress(bit_message)

        message = pack(bit_message, unit_size)
        message = torch.as_tensor(message, dtype=dtype).reshape(shape)
        return message


class PolarCompressSplit:
    high_compress_rate = 0.75
    low_compress_rate = 0.25

    @staticmethod
    def unit_size(dtype):
        if dtype == torch.int8:
            unit_size = 8
        elif dtype == torch.int16:
            unit_size = 16
        else:
            raise NotImplementedError('dtype={}'.format(dtype))
        return unit_size

    @staticmethod
    def compress(message: torch.Tensor):
        unit_size = PolarCompressSplit.unit_size(message.dtype)
        polar_n = unit_size * message.shape.numel()//2
        high_polar_k = int(polar_n * PolarCompressSplit.high_compress_rate)
        low_polar_k = int(polar_n * PolarCompressSplit.low_compress_rate)

        high_rate_message = (message // (2 ** (unit_size // 2)))
        low_rate_message = (message % (2 ** (unit_size // 2)))
        high_bit_message = unpack(high_rate_message.reshape(high_rate_message.shape.numel()), unit_size // 2)
        low_bit_message = unpack(low_rate_message.reshape(low_rate_message.shape.numel()), unit_size // 2)

        high_compressor = PolarCompress(polar_n, high_polar_k)
        low_compressor = PolarCompress(polar_n, low_polar_k)
        return (pack(high_compressor.compress(high_bit_message), 32),
                pack(low_compressor.compress(low_bit_message), 32)), (message.shape, message.dtype)

    @staticmethod
    def decompress(bit_message, ctx):
        shape, dtype = ctx
        unit_size = PolarCompressSplit.unit_size(dtype)
        polar_n = unit_size * shape.numel()//2
        high_bit_message = unpack(bit_message[0], 32)
        low_bit_message = unpack(bit_message[1], 32)
        high_compressor = PolarCompress(polar_n, high_bit_message.shape[0])
        low_compressor = PolarCompress(polar_n, low_bit_message.shape[0])
        high_bit_message = high_compressor.decompress(high_bit_message)
        low_bit_message = low_compressor.decompress(low_bit_message)

        high_rate_message = pack(high_bit_message, unit_size // 2)
        low_rate_message = pack(low_bit_message, unit_size // 2)
        message = high_rate_message * (2 ** (unit_size // 2)) + low_rate_message
        message = torch.as_tensor(message, dtype=dtype).reshape(shape)
        return message
