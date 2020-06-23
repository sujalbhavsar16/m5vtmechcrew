import pandas as pd
import numpy as np
import os



#.........create first group.................
path1 = 'Data2 '
path2 = 'sales_train_evaluation.csv'
sale = pd.read_csv('Data2/sales_train_evaluation.csv')
print(sale)
sale['stecat_id']=sale['state_id']+'_'+sale['cat_id']
sale['stedept_id']=sale['state_id']+'_'+sale['dept_id']
sale['storecat_id']=sale['store_id']+'_'+sale['cat_id']
sale['storedept_id']=sale['store_id']+'_'+sale['dept_id']
sale['itemstore_id']=sale['item_id']+'_'+sale['store_id']
sale['itemstate_id']=sale['item_id']+'_'+sale['state_id']
grpid=['state_id','store_id','cat_id','dept_id','stecat_id','stedept_id','storecat_id','storedept_id','item_id','itemstate_id']
grpdf=pd.DataFrame(columns=np.arange(1,sale.shape[0]+1),index=range(10))
iter=0
for grpid in grpid:
    print(grpid)
    series=sale[grpid]
    # print(set(series))
    # print(sale[bolo].astype('category').cat.codes.astype('int16')+1)
    grpdf.iloc[iter,:]=np.array(sale[grpid].astype('category').cat.codes.astype('int16')+1)
    # print(np.array(sale['state_id'].astype('category').cat.codes.astype('int16')+1).shape)
    iter+=1
# print(grpdf.iloc[0,21342:21345])
# print(set(grpdf.iloc[0]))

grpdf.to_csv('Results/hyndmandf.csv')

print(set(grpdf.iloc[9]))