
import random
import constants as cst

class Lot:
    def __init__(self, i, base_info):
        self.base_info = base_info
        self.lot_class = self.base_info.lot_class
        self.product_type = self.lot_class['Product'].values[i]
        self.step_id = self.lot_class['StepNum'].values[i]
        self.mask = self.lot_class['Mask'].values[i]
        self.pos_machine = self.lot_class[['M' + str(i) for i in cst.MACHINE_LIST]].values[i]

    def __repr__(self):
        return self.product_type


    # lot 생성시 데이터의 확률에 따라 랜덤 생성



