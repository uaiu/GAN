import librosa
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#%%
#filename = 'C:/Users/dbtmd_c3zj9mq/.spyder-py3/hyundai_project/data/mix_wav/Mix_Audio_00000.wav'
filename = 'C:/Users/dbtmd_c3zj9mq/.spyder-py3/hyundai_project/data/clean_wav/Base_Audio_00000.wav'

audio, sr = librosa.load(filename, sr=44100, mono=True)
#%%
class ConvTasNetParam:

    __slots__ = 'causal', 'That', 'C', 'L', 'N', 'B', 'Sc', 'H', 'P', 'X', 'R', 'overlap'

    def __init__(self,
                 causal: bool = False,
                 That: int = 256,
                 C: int = 4,
                 L: int = 16,
                 N: int = 256,
                 B: int = 128,
                 Sc: int = 128,
                 H: int = 256,
                 P: int = 3,
                 X: int = 8,
                 R: int = 3,
                 overlap: int = 8):

        if overlap * 2 > L:
            raise ValueError('`overlap` cannot be greater than half of `L`!')

        self.causal = causal
        self.That = That
        self.C = C
        self.L = L
        self.N = N
        self.B = B
        self.Sc = Sc
        self.H = H
        self.P = P
        self.X = X
        self.R = R
        self.overlap = overlap

    def get_config(self) -> dict:
        return {'causal': self.causal,
                'That': self.That,
                'C': self.C,
                'L': self.L,
                'N': self.N,
                'B': self.B,
                'Sc': self.Sc,
                'H': self.H,
                'P': self.P,
                'X': self.X,
                'R': self.R,
                'overlap': self.overlap}

    def save(self, path: str):
        with open(path, 'w', encoding='utf8') as f:
            f.write('\n'.join(f"{k}={v}" for k, v
                    in self.get_config().items()))

    @staticmethod
    def load(path: str):
        def convert_int(value):
            try:
                return int(value)
            except:
                pass
            return value

        def convert_bool(value):
            if value == 'True':
                return True
            elif value == 'False':
                return False
            else:
                return value

        def convert_tup(tup):
            if tup[0] == 'causal':
                return (tup[0], convert_bool(tup[1]))
            else:
                return (tup[0], convert_int(tup[1]))

        with open(path, 'r', encoding='utf8') as f:
            d = dict(convert_tup(line.strip().split('='))
                     for line in f.readlines())
            return ConvTasNetParam(**d)

    def __str__(self) -> str:
        return f'Conv-TasNet Hyperparameters: {str(self.get_config())}'
#%%
param = ConvTasNetParam()
#%%
num_samples = audio.shape[0]

num_portions = (num_samples - param.overlap) // (param.That * (param.L - param.overlap))

num_samples_output = num_portions * param.That * (param.L - param.overlap)

num_samples = num_samples_output + param.overlap
#%%
audio = audio[:num_samples]
model_input = np.zeros((num_portions, param.That, param.L))
#%%
for i in range(num_portions):
    for j in range(param.That):
        begin = (i * param.That + j) * (param.L - param.overlap)
        end = begin + param.L
        model_input[i][j] = audio[begin:end]
#%%
class Encoder(tf.keras.layers.Layer):

    def __init__(self, param: ConvTasNetParam, **kwargs):
        super(Encoder, self).__init__(name='Encoder', **kwargs)

        self.U = tf.keras.layers.Dense(units=param.N, activation='relu')

    def call(self, mixture_segments):
        # (, That, L) -> (, That, N)
        return self.U(mixture_segments)  # mixture_weights
#%%
encoder = Encoder(param)
mixture_weights = encoder(model_input)
#%%
temp = mixture_weights[0]
#%%
plt.imshow(mixture_weights[0])