class Mask(object):
    """ Class for Mask related objects """

    def __init__(self, base_info):
        self.base_info = base_info
        self.masks = self.base_info.mask
        self.mask_cnt = len(self.base_info.mask)
        self.state = [1] * self.mask_cnt
        self.dup_state = [1] * len(self.base_info.lot_class)

    def initialize(self):
        self.mask_cnt = len(self.base_info.mask)
        self.state = [1] * self.mask_cnt
        self.dup_state = [1] * len(self.base_info.lot_class)
