from datasets.widar import Widar
from utils.conf import data_path_isac


class ISAC(Widar):
    NAME = 'isac'

    def __init__(self, args):
        super(ISAC, self).__init__(args)
        self.DATA_PATH = data_path_isac()
        self.N_SAMPLES_PER_Class = None
        self.N_CLASS = 10