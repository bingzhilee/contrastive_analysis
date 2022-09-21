# coding: utf-8

"""sampling balanced subj-v/aux fixed pattern sentences for representation-extraction procedure"""

import sys
import pandas as pd
from collections import defaultdict
from collections import Counter
import json
import random


file_name = sys.argv[1] # *.tab file containing fixed pattern sents
output = sys.argv[2] # output directory

def group(tab,cat,cat1,cat2):
    dico = dict(iter(tab.groupby(cat)))
    cat_1 = dico[cat1]
    cat_2 = dico[cat2]
    return cat_1,cat_2

tab = pd.read_csv(file_name,sep="\t")
#print(tab[tab["attr"]=='True'].shape) #167
#print(tab[tab["attr"]=='none'].shape) #57
#print(tab[tab['cls_token_num']!=tab['correct_number']].shape)#244
#print(tab[(tab["attr"]!='True')&(tab['correct_number']!=tab['cls_token_num'])].shape)#152

tab = pd.read_csv(file_name,sep="\t")
sing = tab[tab['correct_number']=="sing"]
sing_ids = sing["constr_id"].tolist()
plur = tab[tab['correct_number']=="plur"]
plur_ids = plur["constr_id"].tolist()
print(sing.shape)#516
print(plur.shape)#151
sing_attr = sing[sing["attr"]=='True'] #56
sing_no_attr = sing[sing["attr"]!='True'] #
test_sing_attr = sing_attr.sample(n=50,random_state=1)
test_sing_no_attr = sing_no_attr.sample(n=50,random_state=1)
test_sing = pd.concat([test_sing_attr,test_sing_no_attr])


plur_attr = plur[plur["attr"]=='True'] #110
plur_no_attr = plur[plur["attr"]!='True']
test_plur_attr = plur_attr.sample(n=50,random_state=1)
test_plur_no_attr = plur_no_attr.sample(n=50,random_state=1,replace=True)
test_plur = pd.concat([test_plur_attr,test_plur_no_attr])
# train
train_sing_candidate_ids = [i for i in sing_ids if i not in test_sing["constr_id"].tolist()]
train_plur_candidate_ids = [i for i in plur_ids if i not in test_plur["constr_id"].tolist()]
train_sing_candidate = sing[sing["constr_id"].isin(train_sing_candidate_ids)] #416
train_plur_candidate = plur[plur["constr_id"].isin(train_plur_candidate_ids)] # 69
#breakpoint()
train_sing = train_sing_candidate.sample(n=400,random_state=1)
train_plur = train_plur_candidate.sample(n=400,random_state=1,replace=True)
train = pd.concat([train_sing,train_plur])
test_attr = pd.concat([test_sing_attr,test_plur_attr])
test_no_attr = pd.concat([test_sing_no_attr,test_plur_no_attr])
#breakpoint()
test_attr.to_csv(output + "test_attr.tab", sep="\t", index=False)
test_no_attr.to_csv(output + "test_no_attr.tab", sep="\t", index=False)
train.to_csv(output + "train.tab", sep="\t", index=False)
with open(output + "test_attr.txt","w") as txt:
    txt.write("\n".join(test_attr['sent'].tolist()))
with open(output + "test_no_attr.txt","w") as txt:
    txt.write("\n".join(test_no_attr['sent'].tolist()))
with open(output + "train.txt","w") as txt:
    txt.write("\n".join(train['sent'].tolist()))
