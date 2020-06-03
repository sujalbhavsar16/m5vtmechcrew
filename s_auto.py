from sujallvntcn import s_tcn
from sujallvnbeat import s_nbeat
from functions import get_max_node
import  numpy as np
import pandas as pd
import time
s_atuto=pd.DataFrame(index=['Nbeat','Ntcn'])
case=1
for level in np.arange(1,13):
    for node in range(get_max_node(level)):
        start=time.time()
        print(level,node)
        print('case:',case)
        _,_,rmsse_nbeat=s_nbeat(level=level,node=node,normalized=True,decomposition=True,exo=True,n_in=30,n_out=1)
        _,_,rmsse_ntcn=s_tcn(level=level,node=node,normalized=True,decomposition=True,exo=True,n_in=10,n_out=1)
        s_atuto[str(level)+'_'+str(node)]=[rmsse_nbeat,rmsse_ntcn]
        case+=1
        end=time.time   ()
        ts=end-start
        print('Time spend: %f' %(ts))
