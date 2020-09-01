# -*- coding: UTF-8 -*-
'''
@project:DL_recommend
@author:wangfy
@time:2019/12/5 17:52
'''

import torch
import torch.nn as nn


class Config(object):
    """配置参数"""
    def __init__(self, dataset='', embedding='random'):
        self.model_name = 'ModelMnist'
        # self.train_path = dataset + '/data/train.txt'                                # 训练集
        # self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        # self.test_path = dataset + '/data/test.txt'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        # self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        # self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        # self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 10                         # 类别数
        self.n_vocab = 10000                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率

        self.filter_sizes = (2, 3, 4, 5)                                 # 卷积核尺寸
        self.num_filters = 8                                          # 卷积核数量(channels数)

        self.embed_text = 256
        self.embed_com = 32
        # 用户相关：用户ID个数，性别个数，年龄类别个数，工作个数
        self.uid_max = 6041
        self.gender_max = 2
        self.age_max = 7
        self.job_max = 21

        # 电影相关
        # 电影ID个数
        self.movie_id_max = 3953
        # 电影类型个数
        self.movie_categories_max = 19
        # 电影名单词个数
        self.movie_title_max = 5216




class DecoderUM(nn.Module):
    def __init__(self,config):
        super(DecoderUM, self).__init__()
        self.encoderuser = EncoderUser(config)
        self.encodermoive = EncoderMoive(config)

        self.fcs = nn.Sequential(
            nn.Linear(400,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self, user_inputs, movie_inputs):
#         user_matrix = self.encoderuser(x[:4]) #[B,200]
#         moive_matrix = self.encodermoive(x[4:]) #[B,200]
        user_matrix = self.encoderuser(user_inputs) #[B,200]
#         print(user_matrix.size())
        moive_matrix = self.encodermoive(movie_inputs) #[B,200]
#         print(user_matrix.size())
        um_matrix = torch.cat((user_matrix,moive_matrix),dim=1) #[B,400]
#         print(um_matrix.size())
        scores = self.fcs(um_matrix)  #[10,1]
        return scores,(user_matrix,moive_matrix)

class EncoderUser(nn.Module):
    """
    输入用户的信息，输出用户的特性矩阵
    """
    def __init__(self,config):
        super(EncoderUser, self).__init__()
        self.uid_embedding = nn.Embedding(config.uid_max,config.embed_com)
        self.gender_embedding = nn.Embedding(config.gender_max,config.embed_com//2)
        self.age_embedding = nn.Embedding(config.age_max,config.embed_com//2)
        self.job_embedding = nn.Embedding(config.job_max,config.embed_com//2)

        self.uid_fc = nn.Linear(config.embed_com,config.embed_com)
        self.gender_fc = nn.Linear(config.embed_com//2,config.embed_com)
        self.age_fc = nn.Linear(config.embed_com//2,config.embed_com)
        self.job_fc = nn.Linear(config.embed_com//2,config.embed_com)

        self.combine_fc = nn.Linear(config.embed_com * 4, 200)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x): #[4,B,L] uid,gender,age,job
        x_uid = self.uid_fc(self.uid_embedding(x[0]))
        x_gender = self.gender_fc(self.gender_embedding(x[1]))
        x_age = self.age_fc(self.age_embedding(x[2]))
        x_job = self.job_fc(self.job_embedding(x[3]))

        x_combine = torch.cat((x_uid,x_gender,x_age,x_job),dim=2)
        x_combine = self.relu(x_combine) #应该是要对每一步都进行relu,不过先连接然后relu值不变

        x_combine = self.dropout(x_combine)

        x_user = self.combine_fc(x_combine)
        x_user = self.tanh(x_user) #[1,1,200]
#         print(x_user.size())
        x_user = x_user.squeeze(1)

        return x_user

class EncoderMoive(nn.Module):
    """
    获取电影的特性值
    """
    def __init__(self,config):
        super(EncoderMoive, self).__init__()
        self.encodermoiveid = EncoderMoiveId(config)
        self.encodertitlecnn = EncoderMoiveTitleCNN(config)

        self.fc_moive = nn.Linear(96, 200)

    def forward(self, x): # [3,B,L]
        mid_layer = self.encodermoiveid(x[:2])
        title_layer = self.encodertitlecnn(x[2])

        moive_layer = torch.cat((mid_layer,title_layer),dim=2)
        moive_layer = self.fc_moive(moive_layer)
        moive_layer = torch.tanh(moive_layer).squeeze(1) #[B,200]
        return moive_layer


class EncoderMoiveId(nn.Module):
    """
        输入用户的信息，输出电影的特性矩阵
    """
    def __init__(self,config):
        super(EncoderMoiveId, self).__init__()
        self.mid_embedding = nn.Embedding(config.movie_id_max, config.embed_com)
        self.genre_embedding = nn.Embedding(config.movie_categories_max, config.embed_com)

        self.mid_fc = nn.Linear(config.embed_com, config.embed_com)
        self.genre_fc = nn.Linear(config.embed_com, config.embed_com)


        self.combine_moive_fc = nn.Linear(config.embed_com * 2, 200)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x): #([B,L_id],[B,L_genres])
        x_mid = self.mid_fc(self.mid_embedding(x[0]))
        x_mid = self.relu(x_mid)
        x_genre = self.genre_fc(self.genre_embedding(x[1]))
        x_genre = self.relu(x_genre)

        # 这里简单的对genres向量做一个平均，个人倾向于做一个attention，因为用户可能更加关注某一类型的电影
        x_genre = torch.mean(x_genre,dim=1,keepdim=True)
        x_combine = torch.cat((x_mid,x_genre),dim=-1)

        # x_combine = self.combine_moive_fc(x_combine)
        return x_combine  # [B,1,64]

class EncoderMoiveTitleCNN(nn.Module):
    """
    使用textCNN提取用户的矩阵
    """
    def __init__(self,config):
        super(EncoderMoiveTitleCNN, self).__init__()
        self.title_embedding = nn.Embedding(config.movie_title_max, config.embed_com)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed_com)) for k in config.filter_sizes]
        )
        # self.dropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.5)

    def conv_and_pool(self,x,conv):
        # print("conv_and_pool start ++++ ", x.size())
        x = torch.relu(conv(x)).squeeze(3)
        x = torch.max_pool1d(x,x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # print("forward start ++++ ",x.size())
        out = self.title_embedding(x)
        out = out.unsqueeze(1)  # 扩展到适合使用conv2d，相当于channel_in = 1
        out = torch.cat([self.conv_and_pool(out,conv) for conv in self.convs], 1)
        # print("torch.cat",out.size())
        out = self.dropout(out).unsqueeze(1)  #[B,1,32]
        return out


if __name__ == '__main__':


    config = Config()
    uid = torch.rand((10,1)).long()
    gender = torch.rand((10,1)).long()
    age = torch.rand((10,1)).long()
    job = torch.rand((10,1)).long()
    #
    mid = torch.rand((10,1)).long()
    genres = torch.rand((10,10)).long()
    title = torch.rand((10, 32)).long()
    #
    inputu,inputm = (uid,gender,age,job),(mid,genres,title)
    # # inputu,inputm = (uid,gender,age,job),None
    #
    um_model = DecoderUM(config)
    encoderU = um_model.encoderuser
    print(encoderU)
    um = encoderU(inputu)
    print(um.size())
    # print(um_model)
    # scores,_ = um_model(inputu,inputm)

    # print(scores.size())

    print(sum([p.numel()  for p in um_model.parameters() if p.requires_grad]))


    # title_cnn = EncoderMoiveTitleCNN(config)
    # title_m = title_cnn(input_t)
    # print(title_m.size())

    # input_m = (mid, genres,input_t)
    # encoder_moive = EncoderMoive(config)
    # moive_matrix  = encoder_moive(input_m)
    # print(moive_matrix.size())
    #
    # encoder_user = EncoderUser(config)
    # user_matrix = encoder_user(inputx)
    #
    # print(user_matrix.size())











