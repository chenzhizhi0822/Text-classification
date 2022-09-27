# -*- coding: utf-8 -*-
import pandas as pd
# 中文分词库
import jieba
from sklearn.preprocessing import LabelEncoder  # , label_binarize
# 用于生成TF-IDF特征向量
from sklearn.feature_extraction.text import TfidfVectorizer
# 用于生成测试集评价信息、混淆矩阵、结果正确率
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, \
    precision_recall_fscore_support, roc_curve, precision_recall_curve, auc, recall_score, f1_score, precision_score
import pickle
import re
from sklearn.naive_bayes import GaussianNB
import traceback
import json

# 加载停用词
stopwords = [line.strip() for line in open('stop_words_ch.txt', encoding='utf-8').readlines()]
punctuation = "[.;。/！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝\[\]～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏.]"
punctuation_compiled = re.compile(punctuation)

data_path = '../news_zh/'
train_path = data_path + 'train.txt'
test_path = data_path + 'test.txt'


def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


# 全角转半角、分词，并去除停用词
def cuts(text):
    try:
        text = strQ2B(text)
        text = re.sub(r'\s\s+', '', text)
        text = re.sub(punctuation_compiled, '', text)
        words = list(jieba.cut(text))
        return [word for word in words if word not in stopwords and len(word) != 0]
    except:
        print(traceback.format_exc())
        print(text)


def label2int(label):
    with open(data_path + 'label2int.txt', 'rb') as fin:
        label2int = json.load(fin)
    return label2int.get(label)


def int2label(label_id):
    with open(data_path + 'int2label.txt', 'rb') as fin:
        int2label = json.load(fin)
    return int2label.get(label_id)


# 加载训练集和测试集
df_train = pd.read_table(train_path, header=None, encoding='utf-8')
df_test = pd.read_table(test_path, header=None, encoding='utf-8').dropna()
df_train.columns = ['类别', '文本']
df_test.columns = ['类别', '文本']
print(df_train['类别'].value_counts())

# 数据预处理部分
df_train['文本_cut'] = df_train['文本'].apply(lambda x: ' '.join(cuts(x)))
df_test['文本_cut'] = df_test['文本'].apply(lambda x: ' '.join(cuts(x)))

# 训练TF-IDF模型，用于文本特征提取
vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.01) #TfidfVectorizer可以把原始文本转化为tf-idf的特征矩阵
x_train = vectorizer.fit_transform((d for d in df_train['文本_cut']))
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

x_test = vectorizer.transform((d for d in df_test['文本_cut']))

# 标签数值化
encoder = LabelEncoder()
y_train = encoder.fit_transform(df_train["类别"])

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
y_test = encoder.transform(df_test["类别"])
n_class = len(set(y_train))


# 定义模型效果评估函数，得出每个类别的评价指标
def evaluate(model, x_test, y_test):
    test_mean_accuracy = model.score(x_test.toarray(), y_test)
    y_scores_prob = model.predict_proba(x_test.toarray())
    y_scores = model.predict(x_test.toarray())
    auc = roc_auc_score(y_test, y_scores_prob, multi_class='ovo')
    print(f'test mean accuracy:{test_mean_accuracy}')
    print(f'test auc:{auc}')
    print('报告汇总:')
    print(classification_report(y_test, y_scores))
    return y_scores, y_scores_prob, auc


# 训练
gnb = GaussianNB()
gnb.fit(x_train.toarray(), y_train)
train_score = gnb.score(x_train.toarray(), y_train)
print(f"train mean accuracy: {train_score}")

# 保存模型
with open('bayes.pkl', 'wb') as f:
    pickle.dump(gnb, f)
# 加载模型
with open('bayes.pkl', 'rb') as fr:
    bayes = pickle.load(fr)

# 模型测试
y_scores, y_scores_prob, auc = evaluate(bayes, x_test, y_test)
# 总体正确率和召回率计算
precision = precision_score(y_scores, y_test, average='macro')
accuracy = accuracy_score(y_scores, y_test)
recall = recall_score(y_scores, y_test, average='macro')  # 多分类，二分类是binary
f1 = f1_score(y_scores, y_test, average='macro')
print(f"test set precision:{precision}")
print(f"test set accuracy:{accuracy}")
print(f"test set recall:{recall}")
print(f"test set f1:{f1}")
