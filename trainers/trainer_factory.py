from trainers import *

def build_trainer(option):
    return eval(option.trainer_name)(option)
