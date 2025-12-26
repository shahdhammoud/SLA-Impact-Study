from .gmm_wrapper import GMMWrapper
from .ctgan_wrapper import CTGANWrapper
from .tabddpm_wrapper import TabDDPMWrapper

MODEL_CLASSES = {
    'gmm': GMMWrapper,
    'ctgan': CTGANWrapper,
    'tabddpm': TabDDPMWrapper,
}
