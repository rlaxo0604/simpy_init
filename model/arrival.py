import numpy as np
import random
import constants as cst

class Arrival:
    def __init__(self, base_info):

        self.base_info = base_info
        self.lot_class = self.base_info.lot_class
        self.init_step_target = self.base_info.init_step_target
        self.lot_class_runtime = self.base_info.lot_class_runtime
        self.step_target_runtime = self.base_info.step_target_runtime
        self.steps = self.base_info.steps

    def rand_pick(self):
        """ randomly pick a lot for random arrival and initializer """
        rand = random.random()
        i = -1
        while rand >= 0:
            i += 1
            rand = rand - self.lot_class['Prob'].values[i]

        return i