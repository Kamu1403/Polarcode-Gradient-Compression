import numpy
import numpy as np
import torch
from polarcodes import *
from polar_compression.BSC import AWGN_pure
from polar_compression.Encode import Compress
from polar_compression.Decode import Decompress
from komm._util import pack, unpack
from pytorch_binary_converter import FloatConverter


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


class PolarCompressShorten:
    def __init__(self, N, K, SNR=5.0):
        self.N = N
        self.K = K
        self.SNR = SNR

        # initialise polar code
        self.myPC = PolarCode(self.N, self.K, punct_params=('', 'brs', [], [], None,))
        self.myPC.construction_type = 'bb'

        # mothercode construction
        Shorten(self.myPC, SNR)
        # print(myPC, "\n\n")

    def compress(self, message):
        # set message todo:np or tensor
        full_message = np.zeros(self.myPC.N)
        full_message[np.where(self.myPC.punct_set_lookup != 0)] = message
        AWGN_pure(self.myPC, self.SNR, full_message)
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
        polar_n = unit_size * message.shape.numel() // 2
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
        polar_n = unit_size * shape.numel() // 2
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


class PolarCompressFinal:
    compress_rate = 0.5

    @staticmethod
    def unit_size(dtype):
        if dtype == torch.int8:
            unit_size = 8
        elif dtype == torch.int16:
            unit_size = 16
        elif dtype == torch.int32:
            unit_size = 32
        elif dtype == torch.float16:
            unit_size = 16
        elif dtype == torch.float32:
            unit_size = 32
        else:
            raise NotImplementedError('dtype={}'.format(dtype))
        return unit_size

    @staticmethod
    def to_bits(message: torch.Tensor):
        if message.dtype in [torch.int8, torch.int16, torch.int32]:
            return unpack(message.reshape(message.shape.numel()), PolarCompressFinal.unit_size(message.dtype))
        elif message.dtype in [torch.float16, torch.float32, torch.float64]:
            bit_array = FloatConverter.to_binary(message, PolarCompressFinal.unit_size(message.dtype))
            return bit_array.reshape(bit_array.shape.numel())
        else:
            raise NotImplementedError('dtype={}'.format(message.dtype))

    @staticmethod
    def to_message(bit_message: np.ndarray, ctx):
        shape, dtype = ctx
        if dtype in [torch.int8, torch.int16, torch.int32]:
            message = pack(bit_message, PolarCompressFinal.unit_size(dtype))
            message = torch.as_tensor(message, dtype=dtype).reshape(shape)
            return message
        elif dtype in [torch.float16, torch.float32, torch.float64]:
            unit_size = PolarCompressFinal.unit_size(dtype)
            message = FloatConverter.to_float(
                torch.as_tensor(bit_message.reshape(bit_message.shape[0] // unit_size, unit_size)), unit_size)
            message = torch.as_tensor(message, dtype=dtype).reshape(shape)
            return message
        else:
            raise NotImplementedError('dtype={}'.format(dtype))

    @staticmethod
    def compress(message: torch.Tensor):
        print("Compress message: ", message.shape)
        message = (message * (10 ** 5)).type(torch.int16)

        unit_size = PolarCompressFinal.unit_size(message.dtype)
        polar_n = unit_size * message.shape.numel()
        polar_k = int(polar_n * PolarCompressFinal.compress_rate)
        # bit_message = unpack(message.reshape(message.shape.numel()), unit_size)
        bit_message = PolarCompressFinal.to_bits(message)

        if polar_n == int(2 ** (np.ceil(np.log2(polar_n)))):
            compressor = PolarCompress(polar_n, polar_k)
        else:
            compressor = PolarCompressShorten(polar_n, polar_k)
        return torch.as_tensor(pack(compressor.compress(bit_message), 32)), (message.shape, message.dtype)

    @staticmethod
    def decompress(bit_message, ctx):
        print("Decompress message: ", ctx[0])
        shape, dtype = ctx
        unit_size = PolarCompressFinal.unit_size(dtype)
        polar_n = unit_size * shape.numel()
        bit_message = unpack(bit_message, 32)

        if polar_n == int(2 ** (np.ceil(np.log2(polar_n)))):
            compressor = PolarCompress(polar_n, bit_message.shape[0])
        else:
            compressor = PolarCompressShorten(polar_n, bit_message.shape[0])
        bit_message = compressor.decompress(bit_message)

        # message = pack(bit_message, unit_size)
        # message = torch.as_tensor(message, dtype=dtype).reshape(shape)
        message = PolarCompressFinal.to_message(bit_message, ctx)

        message = message.type(torch.float32) / (10 ** 5)
        return message
