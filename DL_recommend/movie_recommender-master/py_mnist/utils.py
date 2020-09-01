# -*- coding: UTF-8 -*-
'''
@project:DL_recommend
@author:wangfy
@time:2019/12/6 16:13
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
class MovieRankDataset(Dataset):

    def __init__(self, data_org):
        self.dataFrame = data_org

    def __len__(self):
        return len(self.dataFrame)

    def __getitem__(self, idx):
        # user data
        uid = self.dataFrame.ix[idx]['user_id']
        gender = self.dataFrame.ix[idx]['user_gender']
        age = self.dataFrame.ix[idx]['user_age']
        job = self.dataFrame.ix[idx]['user_job']

        # movie data
        mid = self.dataFrame.ix[idx]['movie_id']
        mtype = self.dataFrame.ix[idx]['movie_type']
        mtext = self.dataFrame.ix[idx]['movie_title']

        # target
        rank = torch.FloatTensor([self.dataFrame.ix[idx]['rank']])
        user_inputs = {
            'uid': torch.LongTensor([uid]).view(1, -1),
            'gender': torch.LongTensor([gender]).view(1, -1),
            'age': torch.LongTensor([age]).view(1, -1),
            'job': torch.LongTensor([job]).view(1, -1)
        }

        movie_inputs = {
            'mid': torch.LongTensor([mid]).view(1, -1),
            'mtype': torch.LongTensor(mtype),
            'mtext': torch.LongTensor(mtext)
        }

        sample = {
            'user_inputs': user_inputs,
            'movie_inputs': movie_inputs,
            'target': rank
        }
        return sample

def save_model(model,name=None):
    import time
    import torch
    '''
    保存模型参数
    :param name/path:
    :return: name
    '''

    if name is None:
        prefix = 'checkpoints/' + "model_name" + '_'
        name = time.strftime(prefix + '%m%d_%H_%M_%S.pt')
    torch.save(model.state_dict(), name)
    return name

import logging

class Logger(object):
    def __init__(self, log_file_name, log_level, logger_name):
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file_name)
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] - [%(filename)s line:%(lineno)d] : %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger

