import sys
from subprocess import run

trainDataPath=sys.argv[1]
valDataPath=sys.argv[2]
testDataPath=sys.argv[3]
outputFilePath=sys.argv[4]
questionPart=sys.argv[5]
# subQuestions=['a','b','c','d','e','f']
# for i in range(len(subQuestions)):
try:
    print('\nRunning Q1{}\n'.format(questionPart))
    run(['python3',
    'Q1{}.py'.format(questionPart),
    '{}'.format(trainDataPath),
    '{}'.format(valDataPath),
    '{}'.format(testDataPath),
    '{}'.format(outputFilePath)])
except:
    print(sys.exc_info()[0],end='\n')

