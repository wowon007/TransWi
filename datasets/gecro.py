from datasets.widar import Widar
from utils.conf import data_path_gecro


class GeCro(Widar):
    NAME = 'gecro'

    def __init__(self, args):
        super(GeCro, self).__init__(args)
        self.DATA_PATH = data_path_gecro()
        self.N_SAMPLES_PER_Class = None
        self.N_CLASS = 8