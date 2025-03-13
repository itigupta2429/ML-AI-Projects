#from nltk.corpus import universal_treebanks
from preprocessing import processData

if __name__ == '__main__':
    result = processData('Train.csv')
    print(result)

#[table, chair, computer]
#today, i sat on my chair and turned on the computer. i have a very good computer. 0
#[0, 1, 2]
