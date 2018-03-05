from numpy import *  #和import numpy是一样的比如numpy中random  标准库中也有random，但是两者的功能是不同的，使用from numpy import *容易造成混淆
import operator     #导入运算符模块

def createDataSet():
    group = array ([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]   #读取矩阵第一维的长度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #有问题
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)  #axis=1同一行相加，axis=0同一列相加
    distances = sqDistance**0.5
    sortedDistIndicies = distances.argsort()  #argsort是排序，将元素按照由小到大的顺序返回下标
    classCount = {}        #写成Key-Value形式  定义一个空
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1  #get是取字典里的元素，如果之前这个voteIlabel是有的，那么就返回字典里这个voteIlabel里的值，如果没有就返回0（后面写的），这行代码的意思就是算离目标点距离最近的k个点的类别，这个点是哪个类别哪个类别就加1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #iteritem是一个迭代器，节省空间，需要注意的是在python3中将他改成了“items。operator.itemgetter按照对象中的第几个域中的值进行排序
    return sortedClassCount[0][0]

#分析数据
def file2matrix(filename):    #准备数据，即将文本记录转换为NumPy的解析程序
    fr = open(filename)
    arrayOLines = fr.readlines()    #计算文本文件的行数
    numberOfLines = len(arrayOLines) #
    returnMat = zeros((numberOfLines,3))  #创建返回的数据矩阵 
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()   #去除行的尾部的换行符
        listFromLine = line.split('\t')  #按空格分割 \t表示空格
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))  #取分类标签
        index += 1    #为何索引要自增1？
    return returnMat,classLabelVector

#归一化特征值，减小误差
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges,minVals

#针对分类器的测试代码
def datingClassTest():
    hoRatio = 0.9  #测试集所占的比例
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]   #求出数据的条数
    numTestVecs = int (m*hoRatio)  #求测试集的数据数目
    errorCount = 0.0   #先定义误判的数目
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print ('the classifier came back with: %d, the real answer is: %d ' % (classifierResult,datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print ('the total error rate is: %f' % (errorCount/float(numTestVecs)))
    
#找到某个人并输入他的信息
def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of  time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of icecream consumed per year?'))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 4)
    print ('you will probably like this person:',resultList[classifierResult-1])

#将图像转化为测试向量
def img2Vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range():
    
