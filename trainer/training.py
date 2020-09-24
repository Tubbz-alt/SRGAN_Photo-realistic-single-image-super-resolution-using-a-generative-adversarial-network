import tensorflow as tf
from utils.load_config import LoadConfig
from data_prep.data_processing import DataCreator
from architecture.generator import SRGenerator
from architecture.discriminator import Discriminator
from architecture.load_vgg import VGGModel


# Load Config
config_dict = LoadConfig('./config')()

# Load Dataset
dataset = DataCreator(config_dict).create_data(batch_size=config_dict['batch_size'],
                                               shuffle=False,
                                               check_result=False,
                                               augmentation=False,
                                               save_tf=False)

