# Import Libs
# conda install -c anaconda pandas
from __future__ import division
import pandas as pd
import math
from sklearn.utils import shuffle
from pandas_ml import ConfusionMatrix


# Get CSV
columnNames = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight',
               'classlabel']
train = pd.read_csv("abalone_dataset.txt", delimiter="\t", names=columnNames, header=None)


# For continuous attributes, probability density function for Naive Bayes
def p_x_given_y(x, mean_y, variance_y):
    exponent = float(math.exp(-(math.pow(x - mean_y, 2) / (2 * variance_y))))
    return float((1 / (math.sqrt(2 * math.pi * variance_y))) * exponent)


# Transform Categorical Attributes to Continous Attributes.
# I have used Hot Encoding to transform categorical attribute to binary attributes.
def hotEncodeSex(data, test):
    # Hot Encode Sex for Data and Test
    sex_hot_encode_data = pd.get_dummies(data['Sex'])
    data = data.drop('Sex', axis=1)
    data = data.join(sex_hot_encode_data)
    sex_hot_encode_test = pd.get_dummies(test['Sex'])
    test = test.drop('Sex', axis=1)
    test = test.join(sex_hot_encode_test)

    return data, test


# Find prior probabilities of class label.
# Find mean and variance of the data.
def preCalculations(data):
    num_of1 = data['classlabel'][data['classlabel'] == 1].count()
    num_of2 = data['classlabel'][data['classlabel'] == 2].count()
    num_of3 = data['classlabel'][data['classlabel'] == 3].count()

    total_class_label = data['classlabel'].count()

    prior1 = num_of1 / total_class_label
    prior2 = num_of1 / total_class_label
    prior3 = num_of1 / total_class_label

    data_mean = data.groupby('classlabel').mean()
    data_variance = data.groupby('classlabel').var()

    return prior1, prior2, prior3, data_mean, data_variance


# Calculate the probabilities of the possibility of class label.
# Class label types here are 1, 2 and 3.
# After calculating, find the maximum probability for each of the calculated.
# After that, calculate the accuracy.
# First time I started writing this part, I tried not to use loops for hoping to have better runtime, but it was going
# to be around 400 LOC so converted it to this function which is at O(nˆ3). It's not fast but does the job.
def naiveBayes(attributes, data, test, expected_test):
    data, test = hotEncodeSex(data, test)
    prior1, prior2, prior3, data_mean, data_variance = preCalculations(data)
    priors = [prior1, prior2, prior3]
    class_label_types = [1, 2, 3]
    output = []

    for i in range(len(test)):
        expected_result = expected_test['classlabel'].iloc[i]
        max_of_class_labels = []
        for classLabelType in range(len(class_label_types)):
            probability_of_class_label_type = priors[classLabelType]
            for attribute in range(len(attributes)):
                mean_y = data_mean[attributes[attribute]][data_variance.index == classLabelType + 1].values[0]
                variance_y = \
                    data_variance[attributes[attribute]][
                        data_variance.index == class_label_types[classLabelType]].values[0]
                probability_of_class_label_type = probability_of_class_label_type * p_x_given_y(
                    test[attributes[attribute]].iloc[i], mean_y, variance_y)
            max_of_class_labels.append(probability_of_class_label_type)
        output.append([max_of_class_labels.index(max(max_of_class_labels)) + 1, expected_result])
    return accuracy(output)

# For the following result
def accuracy(result):
    true = 0
    total = len(result)
    cm_expected = []
    cm_predicted = []
    for i in range(len(result)):
        if result[i][0] == result[i][1]:
            true += 1
        cm_expected.append(result[i][1])
        cm_predicted.append(result[i][0])
    misclassified = total - true;
    cm = ConfusionMatrix(cm_expected, cm_predicted)
    cm.print_stats()
    print("----------------------------------------")
    return cm, total, true, misclassified, true/len(result)*100


# Data Preparation for Cases
train = shuffle(train)
case1Data = train.iloc[:100, [0,1,2,8]]
case1Test = train.iloc[101:, [0,1,2]]
case1ExpectedTest = train.iloc[101:, [0,1,2,8]]
case2Data = train.iloc[:1000, [0,1,2,8]]
case2Test = train.iloc[1001:, [0,1,2]]
case2ExpectedTest = train.iloc[1001:, [0,1,2,8]]
case3Data = train.iloc[:2000, [0,1,2,8]]
case3Test = train.iloc[2001:, [0,1,2]]
case3ExpectedTest = train.iloc[2001:, [0,1,2,8]]
case4Data = train.iloc[:100, :]
case4Test = train.iloc[101:, :8]
case4ExpectedTest = train.iloc[101:, :]
case5Data = train.iloc[:1000, :]
case5Test = train.iloc[1001:, :8]
case5ExpectedTest = train.iloc[1001:, :]
case6Data = train.iloc[:2000, :]
case6Test = train.iloc[2001:, :8]
case6ExpectedTest = train.iloc[2001:, :]

usedFeaturesType1 = ['F', 'M', 'I', 'Length', 'Diameter']
usedFeaturesType2 = ['F', 'M', 'I', 'Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']
print("Test 1: ")
naiveBayes(usedFeaturesType1, case1Data, case1Test, case1ExpectedTest)
print("Test 2: ")
naiveBayes(usedFeaturesType1, case2Data, case2Test, case2ExpectedTest)
print("Test 3: ")
naiveBayes(usedFeaturesType1, case3Data, case3Test, case3ExpectedTest)
print("Test 4: ")
naiveBayes(usedFeaturesType2, case4Data, case4Test, case4ExpectedTest)
print("Test 5: ")
naiveBayes(usedFeaturesType2, case5Data, case5Test, case5ExpectedTest)
print("Test 6: ")
naiveBayes(usedFeaturesType2, case6Data, case6Test, case6ExpectedTest)