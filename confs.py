from enum import Enum
import os


class MergeMethod(Enum):
    LINEAR = 'linear'
    GRU_FUSER = 'gru_fuser'
    LSTM_FUSER = 'lstm_fuser'
    FUSER = 'fuser'

class EncoderType(Enum):
    MLP = 'mlp'
    PATCHTST = 'patchtst'

class TSDecoderType(Enum):
    TRANSFORMER = 'transformer'
    LSTM = 'lstm'
    GRU = 'gru'
    ODE_JUMP = 'ode_jump'

class PredictionType(Enum):
    RECONSTRUCTION = ('x',)
    PREDICT = ('x','decoder')
    SIMULATE = ('vae_x','decoder')
    DENOISE = ('noise',)
    PREDICT_EVENT = ('x','decoder','events')

if os.environ.get('dev') == 'true':
    TSDecoderType.FUTURE_GRU = 'future_gru'
