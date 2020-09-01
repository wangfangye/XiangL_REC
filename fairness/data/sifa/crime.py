import json
from collections import Counter
import numpy as np
import matplotlib as mpl

mpl.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号

import matplotlib.pyplot as plt
def init():
    with open(r'D:\code\pycharm\fairness\data\sifa\edu_level.txt', 'r', encoding='UTF-8') as file:
        line = file.readline()
        edu = {}
        eduname = {}
        while line:
            eduname[line.strip()] = len(edu)
            edu[len(edu)] = line.strip()
            line = file.readline()
        file.close()

    with open(r'D:\code\pycharm\fairness\data\sifa\accu.txt', 'r', encoding='UTF-8') as file:
        line = file.readline()
        accu = {}
        accuname = {}
        while line:
            accuname[line.strip()] = len(accu)
            accu[len(accu)] = line.strip()
            line = file.readline()
        file.close()

    return edu, eduname, accu, accuname


edu, eduname, accu, accuname = init()

def dealwithData(inpath, outpath):
    fin = open(inpath, 'r', encoding='UTF-8')
    fout = open(outpath, 'w', encoding='UTF-8')
    # faccu = open('accu.txt', 'a', encoding='UTF-8')
    line = fin.readline()
    while line:
        data = json.loads(line)
        data['meta']['accusation'][0] = data['meta']['accusation'][0].replace(u'罪', '')
        if data['meta']['accusation'][0] in ['非法持有枪支', '非法持枪支']:
            data['meta']['accusation'][0] = '非法持有、私藏枪支、弹药'
        elif data['meta']['accusation'][0] == '骗取贷款':
            data['meta']['accusation'][0] = '骗取贷款、票据承兑、金融票证'
        elif data['meta']['accusation'][0] == '虚开增值税专用发票':
            data['meta']['accusation'][0] = '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票'
        elif data['meta']['accusation'][0] == '非法制造枪支':
            data['meta']['accusation'][0] = '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物'
        elif data['meta']['accusation'][0] in ['运输毒品', '贩卖毒品']:
            data['meta']['accusation'][0] = '走私、贩卖、运输、制造毒品'
        elif data['meta']['accusation'][0] == '拒不执行判决':
            data['meta']['accusation'][0] = '拒不执行判决、裁定'
        elif data['meta']['accusation'][0] == '出售非法制造的发票':
            data['meta']['accusation'][0] = '非法制造、出售非法制造的发票'
        elif data['meta']['accusation'][0] == '销售假药':
            data['meta']['accusation'][0] = '生产、销售假药'
        elif data['meta']['accusation'][0] == '非法制造注册商标标识':
            data['meta']['accusation'][0] = '非法制造、销售非法制造的注册商标标识'
        elif data['meta']['accusation'][0] == '介绍卖淫':
            data['meta']['accusation'][0] = '引诱、容留、介绍卖淫'
        elif data['meta']['accusation'][0] == '非法转让土地使用权':
            data['meta']['accusation'][0] = '非法转让、倒卖土地使用权'
        elif data['meta']['accusation'][0] == '容留卖淫':
            data['meta']['accusation'][0] = '引诱、容留、介绍卖淫'
        elif data['meta']['accusation'][0] == '伪造身份证件':
            data['meta']['accusation'][0] = '伪造、变造居民身份证'
        elif data['meta']['accusation'][0] in ['盗窃依法', '盗窃共同']:
            data['meta']['accusation'][0] = '盗窃'
        elif data['meta']['accusation'][0] == '侵犯公民个人信息':
            data['meta']['accusation'][0] = '非法获取公民个人信息'
        elif data['meta']['accusation'][0] == '故意伤害和寻衅滋事':
            data['meta']['accusation'][0] = '故意伤害'
        elif data['meta']['accusation'][0] == '走私普通货物':
            data['meta']['accusation'][0] = '走私普通货物、物品'
        elif data['meta']['accusation'][0] == '强制猥亵':
            data['meta']['accusation'][0] = '强制猥亵、侮辱妇女'
        elif data['meta']['accusation'][0] == '销售伪劣产品':
            data['meta']['accusation'][0] = '生产、销售伪劣产品'
        elif data['meta']['accusation'][0] == '非法买卖枪支':
            data['meta']['accusation'][0] = '生产、销售伪劣产品'
        elif data['meta']['accusation'][0] == '非法买卖枪支':
            data['meta']['accusation'][0] = '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物'
        elif data['meta']['accusation'][0] in ['伪造公司印章', '伪造事业单位印章']:
            data['meta']['accusation'][0] = '伪造公司、企业、事业单位、人民团体印章'
        elif data['meta']['accusation'][0] == '盗窃其':
            data['meta']['accusation'][0] = '盗窃'
        elif data['meta']['accusation'][0] == '开设赌场共同犯':
            data['meta']['accusation'][0] = '开设赌场'
        elif data['meta']['accusation'][0] in ['包庇', '窝藏']:
            data['meta']['accusation'][0] = '窝藏、包庇'
        elif data['meta']['accusation'][0] == '利用邪教组织破坏法律实施':
            data['meta']['accusation'][0] = '组织、利用会道门、邪教组织、利用迷信破坏法律实施'
        elif data['meta']['accusation'][0] in ['使用虚假身份证件']:
            data['meta']['accusation'][0] = '使用虚假身份证件、盗用身份证件'
        elif data['meta']['accusation'][0] == '盗掘古墓葬':
            data['meta']['accusation'][0] = '盗掘古文化遗址、古墓葬'
        elif data['meta']['accusation'][0] in ['伪造国家机关公文', '伪造国家机关证件', '买卖国家机关证件', '伪造国家机关印章']:
            data['meta']['accusation'][0] = '伪造、变造、买卖国家机关公文、证件、印章'
        elif data['meta']['accusation'][0] == '伪造金融票证':
            data['meta']['accusation'][0] = '伪造、变造金融票证'
        elif data['meta']['accusation'][0] == '危害驾驶':
            data['meta']['accusation'][0] = '危险驾驶'
        elif data['meta']['accusation'][0] == '使用虚假身份证件':
            data['meta']['accusation'][0] = '危险驾驶'
        elif data['meta']['accusation'][0] in ['非法制造弹药', '非法买卖弹药']:
            data['meta']['accusation'][0] = '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物'
        elif data['meta']['accusation'][0] == '诈骗共同犯中的主犯':
            data['meta']['accusation'][0] = '诈骗'
        elif data['meta']['accusation'][0] == '传播淫秽物品牟利':
            data['meta']['accusation'][0] = '制作、复制、出版、贩卖、传播淫秽物品牟利'
        elif data['meta']['accusation'][0] == '聚众扰乱交通秩序':
            data['meta']['accusation'][0] = '聚众扰乱公共场所秩序、交通秩序'
        elif data['meta']['accusation'][0] == '制造毒品':
            data['meta']['accusation'][0] = '走私、贩卖、运输、制造毒品'
        elif data['meta']['accusation'][0] == '妨碍公务':
            data['meta']['accusation'][0] = '妨害公务'
        elif data['meta']['accusation'][0] in ['非法制造爆炸物', '非法储存爆炸物']:
            data['meta']['accusation'][0] = '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物'
        elif data['meta']['accusation'][0] in ['变造身份证件', '买卖身份证件']:
            data['meta']['accusation'][0] = '伪造、变造、买卖身份证件'
        elif data['meta']['accusation'][0] == '单位行贿和虚开增值税专用发票':
            data['meta']['accusation'][0] = '单位行贿'
        elif data['meta']['accusation'][0] == '贩卖淫秽物品牟利':
            data['meta']['accusation'][0] = '制造、贩卖、传播淫秽物品'
        elif data['meta']['accusation'][0] == '故意毁坏他人财物':
            data['meta']['accusation'][0] = '故意毁坏财物'
        elif data['meta']['accusation'][0] in ['走私国家禁止进出口货物', '走私国家禁止进出口的货物']:
            data['meta']['accusation'][0] = '走私国家禁止进出口的货物、物品'
        elif data['meta']['accusation'][0] == '毒品':
            data['meta']['accusation'][0] = '走私、贩卖、运输、制造毒品'
        elif data['meta']['accusation'][0] == '编造虚假恐怖信息':
            data['meta']['accusation'][0] = '编造、故意传播虚假恐怖信息'
        elif data['meta']['accusation'][0] == '走私珍贵动物':
            data['meta']['accusation'][0] = '走私珍贵动物、珍贵动物制品'
        elif data['meta']['accusation'][0] == '传授犯方法':
            data['meta']['accusation'][0] = '传授犯罪方法'
        elif data['meta']['accusation'][0] == '非法处置查封的财产':
            data['meta']['accusation'][0] = '非法处置查封、扣押、冻结的财产'
        elif data['meta']['accusation'][0] == '购买假币':
            data['meta']['accusation'][0] = '出售、购买、运输假币'
        elif data['meta']['accusation'][0] == '非法控制计算机信息系统':
            data['meta']['accusation'][0] = '非法获取计算机信息系统数据、非法控制计算机信息系统'
        elif data['meta']['accusation'][0] == '滥用职权和受贿':
            data['meta']['accusation'][0] = '滥用职权'
        elif data['meta']['accusation'][0] == '涉嫌寻衅滋事':
            data['meta']['accusation'][0] = '寻衅滋事'
        elif data['meta']['accusation'][0] in ['开设赌场分别', '涉嫌开设赌场', '开设赌场和容留他人吸毒']:
            data['meta']['accusation'][0] = '开设赌场'
        elif data['meta']['accusation'][0] == '销售不符合安全标准的产品':
            data['meta']['accusation'][0] = '生产、销售不符合安全标准的食品'
        elif data['meta']['accusation'][0] == '非法储存弹药':
            data['meta']['accusation'][0] = '非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物'
        elif data['meta']['accusation'][0] == '非法倒卖土地使用权':
            data['meta']['accusation'][0] = '非法转让、倒卖土地使用权'
        elif data['meta']['accusation'][0] == '容留吸毒':
            data['meta']['accusation'][0] = '容留他人吸毒'
        elif data['meta']['accusation'][0] == '交通肇事罪':
            data['meta']['accusation'][0] = '交通肇事'
        elif data['meta']['accusation'][0] not in accuname:
            line = fin.readline()
            continue
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
        line = fin.readline()
    fin.close()
    fout.close()


def \
        getData(path):
    """
    :parameter:
    :return: allFact, accu_label, allCrimalInfo{'age','edu_level','race','gender'}

    age<='25' : 0 , '25'<age<='45': 1, age > '45' : 2
    race == '汉'?  0 : 1
    gender == '男'? 0 : 1
    """
    with open(path, 'r', encoding='UTF-8') as f:
        line = f.readline()
        allFact = []
        allCrimalInfo = []
        accu_label = []
        accu_ra=[]
        while line:
            data = json.loads(line)
            if data['crimal_information']['age'] <= '18' : age = 0
            elif '18' < data['crimal_information']['age'] <= '35' : age = 1
            elif '35' < data['crimal_information']['age'] <= '50' : age = 2
            else: age = 3

            e = data['crimal_information']['edu_level']
            if '博士' in e or '硕士' in e or '研究生' in e:
                edu_level = 0
            elif ('大学' in e or '本科' in e) and '专科' not in e :
                edu_level = 1
            elif '专科' in e or '大专' in e :
                edu_level = 2
            elif '中专' in e or '中技' in e or '职高' in e or '职校' in e or '高职' in e or '中等' in e :
                edu_level = 3
            elif '技工' in e or '技校' in e :
                edu_level = 4
            elif '高中' in e :
                edu_level = 5
            elif '初中' in e :
                edu_level = 6
            elif '小学' in e :
                edu_level = 7
            elif '文盲' in e or '文' in e or '半' in e:
                edu_level = 8
            else:
                line = f.readline()
                continue

            if data['crimal_information']['race'] == '汉': race = 0
            else: race = 1

            if data['crimal_information']['gender'] == '男': gender = 0
            else: gender = 1

            allFact.append(data['fact'])
            allCrimalInfo.append([age, edu_level, race, gender])
            accu_label.append(accuname[data['meta']['accusation'][0]])
            accu_ra.append(data['meta']['relevant_articles'])
            line = f.readline()

        f.close()
    return allFact, allCrimalInfo, accu_label,accu_ra

def accuFig(accu_label):
    accuDict = {}
    # get unique labels
    for label in set(accu_label):
        accuDict[label] = accu_label.count(label)
    most_common = Counter(accuDict).most_common(20)
    for i in range(len(most_common)):
        print(most_common[i][0], accu[most_common[i][0]], most_common[i][1])
    x = accuDict.keys()
    y = accuDict.values()
    plt.bar(x, y)
    plt.show()


def attriFig(crilabel, allCrimalInfo, accu_label):
    """
    查看某一罪名，犯罪嫌疑人不同特征取值的分布
    :param crilabel: 特定罪名
    :param attri: 特征数组
    :param allCrimalInfo: 罪犯信息
    :param accu_label: 所有涉及罪名
    :return:
    """
    index = [i for i in range(len(accu_label)) if accu_label[i]==crilabel]
    allCrimalInfoArr = np.array(allCrimalInfo)
    cInfo = allCrimalInfoArr[np.ix_(index)]
    print(cInfo, len(cInfo))
    age_label = [0, 1, 2,3]
    edu_label = edu.keys()
    race_label = [0, 1]
    gender_label = [0, 1]
    ageSet = {}; eduSet = {}; raceSet = {}; genderSet = {}
    for age in age_label:
        ageSet[age] = cInfo[:, 0].tolist().count(age)
    for e in edu_label:
        print(e,cInfo[:, 1].tolist().count(e))
        eduSet[e] = cInfo[:, 1].tolist().count(e)
    for race in race_label:
        raceSet[race] = cInfo[:, 2].tolist().count(race)
    for gender in gender_label:
        genderSet[gender] = cInfo[:, 2].tolist().count(gender)
    fig, axs = plt.subplots(2, 2, figsize=(8,8))
    axs[0,0].bar(['less than 25', '18-35','35-50', 'older than 50'], ageSet.values(), width=0.5)
    axs[0,0].set_title('Age')
    axs[0,1].bar(edu.keys(), eduSet.values())
    axs[0,1].set_title('Edu_level')
    axs[1,0].bar(['汉', '其他'], raceSet.values())
    axs[1,0].set_title('Race')
    axs[1,1].bar(['male', 'female'], genderSet.values())
    axs[1,1].set_title('Gender')
    # fig.suptitle(accu[crilabel])
    plt.show()

def test_orig():
    dealwithData(r'D:\code\pycharm\fairness\data\raw\law\data_2.json', r'D:\code\pycharm\fairness\data\raw\law\Crime_data2.json')
    # allFact, allCrimalInfo, accu_label = getData(r'D:\code\pycharm\fairness\data\raw\law\data.json')
    # accuFig(accu_label)
    #
    # print(allFact, '\n', allCrimalInfo, '\n', accu_label)
    # attriFig(190, allCrimalInfo, accu_label)

if __name__== "__main__":
    # test_orig()
    # allFact, allCrimalInfo, accu_label = getData('crimeData.json')
    allFact, allCrimalInfo, accu_label, accu_ra = getData(r'D:\code\pycharm\fairness\data\raw\law\Crime_data2.json')
    # print(len(allFact))
    # print(len(allCrimalInfo))
    # print(len(accu_label))
    # print(len(accu_ra))
    # print(accu_ra[:10])
    # dealwithData('data.json', 'crimeData.json')
    # allFact, allCrimalInfo, accu_label = getData('crimeData.json')
    accuFig(accu_label)

    # print(allFact, '\n', allCrimalInfo, '\n', accu_label)
    attriFig(152, allCrimalInfo, accu_label)