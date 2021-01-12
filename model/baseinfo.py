import pandas as pd
import constants as cst


class BaseInfo:
    def __init__(self):
        """
        Base information for photo simulator, like lot class, initial step target, and steps.
        :param version_id: TRAIN version_id is used for matching the lot class for RUNTIME.
        """
        if cst.DATA_TYPE == 'xlsx':
            self.lot_class = pd.read_excel(cst.DATA_SOURCE + '/Arrival.xlsx')  # cluster
            self.init_step_target = pd.read_excel(cst.DATA_SOURCE + '/StepTarget.xlsx')
            self.lot_class_runtime = pd.read_csv(cst.DATA_SOURCE + '/Arrival_runtime.txt')
            self.step_target_runtime = pd.read_csv(cst.DATA_SOURCE + '/StepTarget_runtime.txt')

            self.mask = pd.read_excel(cst.DATA_SOURCE + '/MASK.xlsx')

            self.dedication = pd.read_excel(cst.DATA_SOURCE + '/Dedication.xlsx')
            self.lot_class_m = {}
            for m in cst.MACHINE_LIST:
                self.lot_class_m[m] = pd.read_excel(cst.DATA_SOURCE + '/Arrival_' + str(m) + '.xlsx')  # cluster

        elif cst.DATA_TYPE == 'txt':
            self.lot_class = pd.read_csv(cst.DATA_SOURCE + '/Arrival.txt')
            self.init_step_target = pd.read_csv(cst.DATA_SOURCE + '/StepTarget.txt')
            self.lot_class_runtime = pd.read_csv(cst.DATA_SOURCE + '/Arrival_runtime.txt')
            self.step_target_runtime = pd.read_csv(cst.DATA_SOURCE + '/StepTarget_runtime.txt')

            self.mask = pd.read_csv(cst.DATA_SOURCE + '/MASK.txt')

            self.dedication = pd.read_csv(cst.DATA_SOURCE + '/Dedication.txt')

            self.lot_class_m = {}
            for m in cst.MACHINE_LIST:
                self.lot_class_m[m] = pd.read_csv(cst.DATA_SOURCE + '/Arrival_' + str(m) + '.txt')

        self.steps = list(self.lot_class.drop_duplicates('StepNum')['StepNum'])
