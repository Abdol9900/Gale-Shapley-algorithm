import random
import json 

data = {}
dictionaryKeyLength = 50
arrayListLength = 50

		
def shuffleArray(length):
    baseList = []
    for index in range(length):
        baseList.append(str(index + 1))

    random.shuffle(baseList)
    return baseList

for index in range(dictionaryKeyLength):
    data[str(index + 1)] = shuffleArray(arrayListLength)

with open('./women_list.txt' , 'w') as f: f.write(json.dumps(data)) # saved nummbers in women_list.txt 