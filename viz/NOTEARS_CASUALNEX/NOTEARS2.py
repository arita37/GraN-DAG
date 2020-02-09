
# coding: utf-8

# In[1]:


import os,sys; print(os)
sys.path.append('D:/_devs/Python01/gitdev/aapackage/aapackage/')


"""


from util_session  import *

save_session( "causalnex_credit", glob= globals() )


x= 5
str(type(globals().get(x) ))



"""


"""
!pip install pgmpy
!pip install sklearn pandas tqdm funcsigs statsmodels community packaging
!pip install torch
!pip install bnlearn



LIMIT_BAL	SEX	EDUCATION	MARRIAGE	AGE	PAY_0	PAY_2	PAY_3	PAY_4	PAY_5	PAY_6	BILL_AMT1	BILL_AMT2	BILL_AMT3	BILL_AMT4	BILL_AMT5	BILL_AMT6	PAY_AMT1	PAY_AMT2	PAY_AMT3	PAY_AMT4	PAY_AMT5	PAY_AMT6	default.payment.next.month




"""


# In[ ]:

from causalnex.structure.notears import from_pandas
import pandas as pd
import numpy as np


# In[ ]:

# !pip install causalnex


# In[15]:
df = pd.read_csv('creditdata.csv', delimiter=',')
df = df.set_index("ID")
coldf = list( df.columns )


colnum = [  'AGE', 'BILL_AMT1',  'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
            'BILL_AMT5', 'BILL_AMT6',  'PAY_AMT1',  'PAY_AMT2', 'PAY_AMT3',  'PAY_AMT4', 'PAY_AMT5',  'PAY_AMT6']
coly = ['y']


df = df [colnum + coly ]

df.head(5)

data = df 



# In[16]:
dfs = df.copy()

non_numeric_columns = list(dfs.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in non_numeric_columns:
    dfs[col] = le.fit_transform(dfs[col])

dfs.head(5)


####################################################################################################
###### Pre-processing  #############################################################################
for x in colnum :
    df[x] = np.log( df[x] + 0.1 )


for x in colnum :
    df[x] = df[x].fillna( df[x].median() )







# In[17]
dfs = df.copy()
from causalnex.structure.notears import from_pandas
sm = from_pandas(dfs.iloc[:1000, :])



# In[18]:
import copy
sm0 = copy.deepcopy( sm )




# In[19]:

from causalnex.plots import plot_structure
_, _, _ = plot_structure(sm)



sm =  copy.deepcopy( sm0 )


sm.remove_edges_below_threshold(0.2)
_, _, _ = plot_structure(sm)




# In[21]:

sm = sm.get_largest_subgraph()

_, _, _ = plot_structure(sm)


# In[20]:

from causalnex.network import BayesianNetwork
bn = BayesianNetwork(sm)






dfbin = df.copy()


for x in colnum :
    m = dfbin[x].median()
    dfbin[x] = dfbin[x].apply( lambda  t : 1 if t > m else 0  )






# In[36]:

# Split 90% train and 10% test
from sklearn.model_selection import train_test_split
train, test = train_test_split(dfbin, train_size=0.9, test_size=0.1, random_state=7)



# In[37]:

"""

Model Probability
With the learnt structure model from earlier and the discretised data, we can now fit the probability distrbution of the Bayesian Network. The first step in this is specifying all of the states that each node can take. This can be done either from data, or providing a dictionary of node values. We use the full dataset here to avoid cases where states in our test set do not exist in the training set. For real-world applications, these states may need to be provided using the dictionary method.


"""


bn = bn.fit_node_states( dfbin )




# In[38]:


# Fit Conditional Probability Distributions


bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

ptable = bn.cpds


ptable['PAY_AMT6']


pred = bn.predict(dfbin, 'y')


from causalnex.evaluation import classification_report
classification_report(bn, test, "y")



from causalnex.evaluation import roc_auc
roc, auc = roc_auc(bn, test, "y")
print(auc)




#### Marginals after Observations
from causalnex.inference import InferenceEngine
ie = InferenceEngine(bn)
marginals = ie.query()
marginals["y"]# In[ ]:



import numpy as np
labels, counts = np.unique(dfbin["y"], return_counts=True)
list(zip(labels, counts))





##### Do Calculus  ###############################################################
print("distribution before do", ie.query()["higher"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})

print("distribution after do", ie.query()["higher"])


ie.reset_do("higher")


print("marginal G1", ie.query()["G1"])
ie.do_intervention("higher",
                   {'yes': 1.0,
                    'no': 0.0})
print("updated marginal G1", ie.query()["G1"])






# In[ ]:




# In[ ]:




# In[ ]:



