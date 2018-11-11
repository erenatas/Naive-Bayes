import math

import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle

# Get CSV
columnNames=['Sex','Length','Diameter','Height','Whole weight','Shucked weight','Viscera weight','Shell weight','classlabel']
train = pd.read_csv("abalone_dataset.txt",delimiter="\t", names=columnNames, header=None)

# Data Preparation for Cases
train = shuffle(train)
case1Data = train.iloc[:100, [0,1,2,8]]
case1Test = train.iloc[101:, [0,1,2]]
case1ExpectedTest = train.iloc[101:, [0,1,2,8]]
train = shuffle(train)
case2Data = train.iloc[:1000, [0,1,2,8]]
case2Test = train.iloc[1001:, [0,1,2]]
train = shuffle(train)
case3Data = train.iloc[:2000, [0,1,2,8]]
case3Test = train.iloc[2001:, [0,1,2,8]]
train = shuffle(train)
case4Data = train.iloc[:100, :]
case4Test = train.iloc[101:, :8]
train = shuffle(train)
case5Data = train.iloc[:1000, :]
case5Test = train.iloc[1001:, :8]
train = shuffle(train)
case6Data = train.iloc[:2000, :]
case6Test = train.iloc[2001:, :8]


def type1(data, test, expectedTest):
    # Hot Encode Sex for Data and Test
    sexHotEncodeData = pd.get_dummies(data['Sex'])
    data = data.drop('Sex', axis=1)
    data = data.join(sexHotEncodeData)
    sexHotEncodeTest = pd.get_dummies(test['Sex'])
    test = test.drop('Sex', axis=1)
    test = test.join(sexHotEncodeTest)

    numOf1 = data['classlabel'][data['classlabel'] == 1].count()
    numOf2 = data['classlabel'][data['classlabel'] == 2].count()
    numOf3 = data['classlabel'][data['classlabel'] == 3].count()

    totalClassLabel = data['classlabel'].count()

    prior1 = numOf1 / totalClassLabel
    prior2 = numOf1 / totalClassLabel
    prior3 = numOf1 / totalClassLabel

    dataMean = data.groupby('classlabel').mean()
    dataVariance = data.groupby('classlabel').var()

    # Means for 1,2,3
    oneFMean = dataMean['F'][dataVariance.index == 1].values[0]
    oneIMean = dataMean['I'][dataVariance.index == 1].values[0]
    oneMMean = dataMean['M'][dataVariance.index == 1].values[0]
    oneLengthMean = dataMean['Length'][dataVariance.index == 1].values[0]
    oneDiameterMean = dataMean['Diameter'][dataVariance.index == 1].values[0]

    twoFMean = dataMean['F'][dataVariance.index == 2].values[0]
    twoIMean = dataMean['I'][dataVariance.index == 2].values[0]
    twoMMean = dataMean['M'][dataVariance.index == 2].values[0]
    twoLengthMean = dataMean['Length'][dataVariance.index == 2].values[0]
    twoDiameterMean = dataMean['Diameter'][dataVariance.index == 2].values[0]

    threeFMean = dataMean['F'][dataVariance.index == 3].values[0]
    threeIMean = dataMean['I'][dataVariance.index == 3].values[0]
    threeMMean = dataMean['M'][dataVariance.index == 3].values[0]
    threeLengthMean = dataMean['Length'][dataVariance.index == 3].values[0]
    threeDiameterMean = dataMean['Diameter'][dataVariance.index == 3].values[0]

    # Variances for 1,2,3
    oneFVariance = dataVariance['F'][dataVariance.index == 1].values[0]
    oneIVariance = dataVariance['I'][dataVariance.index == 1].values[0]
    oneMVariance = dataVariance['M'][dataVariance.index == 1].values[0]
    oneLengthVariance = dataVariance['Length'][dataVariance.index == 1].values[0]
    oneDiameterVariance = dataVariance['Diameter'][dataVariance.index == 1].values[0]

    twoFVariance = dataVariance['F'][dataVariance.index == 2].values[0]
    twoIVariance = dataVariance['I'][dataVariance.index == 2].values[0]
    twoMVariance = dataVariance['M'][dataVariance.index == 2].values[0]
    twoLengthVariance = dataVariance['Length'][dataVariance.index == 2].values[0]
    twoDiameterVariance = dataVariance['Diameter'][dataVariance.index == 2].values[0]

    threeFVariance = dataVariance['F'][dataVariance.index == 3].values[0]
    threeIVariance = dataVariance['I'][dataVariance.index == 3].values[0]
    threeMVariance = dataVariance['M'][dataVariance.index == 3].values[0]
    threeLengthVariance = dataVariance['Length'][dataVariance.index == 3].values[0]
    threeDiameterVariance = dataVariance['Diameter'][dataVariance.index == 3].values[0]

    result = []

    for i in range(len(test)):
        try:
            probabilityOf1 = prior1 * \
                             p_x_given_y(test['F'][i], oneFMean, oneFVariance) * \
                             p_x_given_y(test['I'][i], oneIMean, oneIVariance) * \
                             p_x_given_y(test['M'][i], oneMMean, oneMVariance) * \
                             p_x_given_y(test['Length'][i], oneLengthMean, oneLengthVariance) * \
                             p_x_given_y(test['Diameter'][i], oneDiameterMean, oneDiameterVariance)

            probabilityOf2 = prior2 * \
                             p_x_given_y(test['F'][i], twoFMean, twoFVariance) * \
                             p_x_given_y(test['I'][i], twoIMean, twoIVariance) * \
                             p_x_given_y(test['M'][i], twoMMean, twoMVariance) * \
                             p_x_given_y(test['Length'][i], twoLengthMean, twoLengthVariance) * \
                             p_x_given_y(test['Diameter'][i], twoDiameterMean, twoDiameterVariance)

            probabilityOf3 = prior3 * \
                             p_x_given_y(test['F'][i], threeFMean, threeFVariance) * \
                             p_x_given_y(test['I'][i], threeIMean, threeIVariance) * \
                             p_x_given_y(test['M'][i], threeMMean, threeMVariance) * \
                             p_x_given_y(test['Length'][i], threeLengthMean, threeLengthVariance) * \
                             p_x_given_y(test['Diameter'][i], threeDiameterMean, threeDiameterVariance)

            results = [probabilityOf1, probabilityOf2, probabilityOf3]
            minimumValue = min(results)
            if minimumValue == probabilityOf1:
                output = 1
            if minimumValue == probabilityOf2:
                output = 2
            if minimumValue == probabilityOf3:
                output = 3

            expectedResult = expectedTest['classlabel'][i]

            result.append([output, expectedResult])
        except:
            print(i, test['F'][i], oneFMean, oneFVariance, oneIMean, oneIVariance, oneMMean, oneMVariance)

    return result


def p_x_given_y(x, mean_y, variance_y):
    # Input the arguments into a probability density function
    exponent = math.exp(-(math.pow(x - mean_y, 2) / (2 * variance_y)))
    return (1 / (math.sqrt(2 * math.pi * variance_y))) * exponent


test = case1Test
sexHotEncodeTest = pd.get_dummies(test['Sex'])
test = test.drop('Sex', axis = 1)
test = test.join(sexHotEncodeTest)

type1(case1Data, case1Test, case1ExpectedTest)