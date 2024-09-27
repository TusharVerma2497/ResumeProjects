import sys
from subprocess import run

trainDataPath=sys.argv[1]
testDataPath=sys.argv[2]
outputFilePath=sys.argv[3]
questionPart=sys.argv[4]
# subQuestions=['a','b','c','d','e','f']
# for i in range(len(subQuestions)):
try:
    print('\nRunning Q2{}\n'.format(questionPart))
    run(['python3',
    'Q2{}.py'.format(questionPart),
    '{}'.format(trainDataPath),
    '{}'.format(testDataPath),
    '{}'.format(outputFilePath)])
except:
    print(sys.exc_info()[0],end='\n')

