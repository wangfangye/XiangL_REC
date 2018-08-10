# -*- coding:utf-8 -*-  

import urllib
import json

# 注意这里用unicode编码，否则会显示乱码
content = input("请输入要翻译的内容：")
# 网址是Fig6中的 Response URL
url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=http://www.youdao.com/'
# 爬下来的数据 data格式是Fig7中的 Form Data
data = {}
data['type'] = 'AUTO'

data['i'] = content
data['doctype'] = 'json'
data['xmlVersion'] = '1.6'
data['keyfrom'] = 'fanyi.web'
data['ue'] = 'UTF-8'
data['typoResult'] = 'true'

# 数据编码
data = urllib.urlencode(data)

# 按照data的格式从url爬内容
response = urllib.urlopen(url, data)
# 将爬到的内容读出到变量字符串html，
html = response.read()
# 将字符串转换成Fig8所示的字典形式
target = json.loads(html)
# 根据Fig8的格式，取出最终的翻译结果
result = target["translateResult"][0][0]['tgt']

# 这里用unicode显示中文，避免乱码
print(u"翻译结果：%s" % (target["translateResult"][0][0]['tgt']))