import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, BertTokenizerFast
import os
import json
import argparse
import pickle

# Defining some key variables that will be used later on in the training
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
train_size = 0.8 #fraction of the total dataset being used for training

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

"""
This makes a directory with the basename of path with a number appended to the
end. If such a directory already exists, the number will be increased.

ex. path/to/some/file.txt produces a directory file0. If file0 already exists
then file1 will be created.

path is a string representing a file path.
"""
def makeUniqueDirectory(path, location = os.getcwd()):
    base = os.path.basename(path).split(".")[0]
    pathInUse = True
    count = 0
    while pathInUse:
        savePath=os.path.join(location,f"{base}_{count}")
        if os.path.exists(savePath):
            count +=1
        else:
            pathInUse = False
            os.mkdir(savePath)

    return savePath

if __name__ == '__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument("-p", "--path", dest="path", help="path to json file if starting new training, or model directory if restarting")
    # parse.add_argument("-r", "--restart", dest="restart", action='store_true', help="flag which indicates to restart a training procedure.")
    args = parse.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Could not find given path: {args.path}")
    
    savePath = args.path

    with open(os.path.join(savePath,"vectormap.txt"),"r") as f:
        vectorMap = eval(f.read())

    with open(os.path.join(savePath,"dataLocation.txt"),"r") as f:
        dataPath = f.read()

    # with open(os.path.join(savePath,"completedEpochs.txt"),"r") as f:
    #     completedEpochs = int(f.read())

    print("Importing raw abstracts json")
    with open(os.path.join(os.getcwd(), dataPath),"r", encoding='utf_8') as file:
        abstractsRawJSON = file.read()
    rawJSON = json.loads(abstractsRawJSON)
    abstracts = rawJSON['events']
    divisions = rawJSON['tags']

    tags = list(divisions.keys())
    nTags = len(tags)

data = []
for i in range(len(abstracts)):
    tagVector = [0 for i in range(nTags)]
    for tag in abstracts[i]['tags']:
        tagVector[vectorMap[tag]] = 1
    
    data.append([abstracts[i]['abstract'],tagVector])

formattedData = pd.DataFrame (data,columns=['abstract','list'])

print("Checking if cuda gpu is available.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Proceeding using {device}")

print("initializing dataset class.")
tokenizerpath = os.path.join(savePath)
tokenizer = BertTokenizerFast.from_pretrained(tokenizerpath)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.abstract = dataframe.abstract
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.abstract)

    def __getitem__(self, index):
        comment_text = str(self.abstract[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

print("Splitting the data into training and testing sets.")
train_dataset=formattedData.sample(frac=train_size,random_state=200) #This should be repeatable provided the json order doesn't change
test_dataset=formattedData.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(formattedData.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

print("formatting data using dataset class.")
training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

print("Formatting data using torch data loader")
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

print("Initializing BERT NN class.")
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, nTags)
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

print("Initializing BERT NN and sending to device.")
modelpath = os.path.join(savePath,"model.bin")
model = torch.load(modelpath, map_location=device)

# model = BERTClass()
# model.to(device)

print("Defining validation routine.")
def validation(epoch):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

validationDir = makeUniqueDirectory("metrics",location=savePath)

print("Validating")
accuracies = []
f1micros = []
f1macros = []
confusions = []
for epoch in range(EPOCHS):
    outputs, targets = validation(epoch)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print("outputs")
    print(outputs)
    print("targets")
    print(targets)
    confusionMatrix = metrics.multilabel_confusion_matrix(targets, outputs)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    print(f"Confusion Matrix\n{confusionMatrix}")

    accuracies.append(accuracy)
    f1micros.append(f1_score_micro)
    f1macros.append(f1_score_macro)
    confusions.append(confusionMatrix)

with open(os.path.join(validationDir,"accuracy.txt"),"w") as f:
    f.write(str(accuracies))

with open(os.path.join(validationDir,"f1micro.txt"),"w") as f:
    f.write(str(f1micros))

with open(os.path.join(validationDir,"f1macro.txt"),"w") as f:
    f.write(str(f1macros))

with open(os.path.join(validationDir,"confusions.pickle"), mode='wb') as f:
        pickle.dump(confusions,f)