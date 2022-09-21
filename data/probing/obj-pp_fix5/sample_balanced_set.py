# coding: utf-8

"""sampling balanced obj-pp fixed pattern sentences for representation-extraction procedure"""

import sys
import pandas as pd
from collections import defaultdict
from collections import Counter
import json
import random


file_name = sys.argv[1] # *.tab file containing fixed pattern sents
output = sys.argv[2] # output directory


tab = pd.read_csv(file_name,sep="\t")
sing = tab[tab['correct_number']=="sing"]

sing_ids = sing["constr_id"].tolist()
plur = tab[tab['correct_number']=="plur"]
plur_ids = plur["constr_id"].tolist()
print(sing.shape)
print(plur.shape)
sing_attr = sing[sing['cls_noun_num']!=sing['correct_number']] #142
sing_no_attr = sing[sing['cls_noun_num']==sing['correct_number']] #142
test_sing_attr = sing_attr.sample(n=50,random_state=20)
test_sing_no_attr = sing_no_attr.sample(n=50,random_state=20)
test_sing = pd.concat([test_sing_attr,test_sing_no_attr])


plur_attr = plur[plur['cls_noun_num']!=plur['correct_number']]
plur_no_attr = plur[plur['cls_noun_num']==plur['correct_number']]
test_plur_attr = plur_attr.sample(n=50,random_state=20)
test_plur_no_attr = plur_no_attr.sample(n=50,random_state=20)
test_plur = pd.concat([test_plur_attr,test_plur_no_attr])
# train
train_sing_candidate_ids = [i for i in sing_ids if i not in test_sing["constr_id"].tolist()]
train_plur_candidate_ids = [i for i in plur_ids if i not in test_plur["constr_id"].tolist()]
train_sing_candidate = sing[sing["constr_id"].isin(train_sing_candidate_ids)]
train_plur_candidate = plur[plur["constr_id"].isin(train_plur_candidate_ids)]
train_sing = train_sing_candidate.sample(n=400,random_state=20)
train_plur = train_plur_candidate.sample(n=400,random_state=20)
train = pd.concat([train_sing,train_plur])
test_attr = pd.concat([test_sing_attr,test_plur_attr])
test_no_attr = pd.concat([test_sing_no_attr,test_plur_no_attr])

test_attr.to_csv(output + "test_attr.tab", sep="\t", index=False)
test_no_attr.to_csv(output + "test_no_attr.tab", sep="\t", index=False)
train.to_csv(output + "train.tab", sep="\t", index=False)
with open(output + "test_attr.txt","w") as txt:
    txt.write("\n".join(test_attr['sent'].tolist()))
with open(output + "test_no_attr.txt","w") as txt:
    txt.write("\n".join(test_no_attr['sent'].tolist()))
with open(output + "train.txt","w") as txt:
    txt.write("\n".join(train['sent'].tolist()))