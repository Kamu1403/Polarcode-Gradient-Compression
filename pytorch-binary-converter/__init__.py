from binary_converter import float2bit, bit2float


class FloatConverter:
    config = {
        64: {
            'precision': 'double',
            'exp': 11,
            'mant': 52,
            'bias': 1023.,
        },
        32: {
            'precision': 'single',
            'exp': 8,
            'mant': 23,
            'bias': 127.,
        },
        16: {
            'precision': 'half',
            'exp': 5,
            'mant': 10,
            'bias': 15.,
        }
    }

    @staticmethod
    def to_binary(tensor, bits):
        conf = FloatConverter.config[bits]
        float2bit(tensor, num_e_bits=conf['exp'], num_m_bits=conf['mant'], bias=conf['bias'])

    @staticmethod
    def to_float(tensor, bits):
        conf = FloatConverter.config[bits]
        bit2float(tensor, num_e_bits=conf['exp'], num_m_bits=conf['mant'], bias=conf['bias'])
