
# coding: utf-8

# In[1]:


import os,sys; print(os)
sys.path.append('D:/_devs/Python01/gitdev/aapackage/aapackage/')
"""


from util_session  import *

save_session( "causalnex_test", glob= globals() )


x= 5
str(type(globals().get(x) ))



"""


"""
!pip install pgmpy
!pip install sklearn pandas tqdm funcsigs statsmodels community packaging
!pip install torch
!pip install bnlearn

"""


# In[ ]:

from causalnex.structure.notears import from_pandas
import pandas as pd
import numpy as np


# In[ ]:

# !pip install causalnex


# In[15]:

import pandas as pd
df = pd.read_csv('student-por.csv', delimiter=';')
df.head(5)


drop_col = ['school','sex','age','Mjob', 'Fjob','reason','guardian']
df = df.drop(columns=drop_col)
df.head(5)

data =df 


# In[16]:

import numpy as np
dfs = df.copy()

non_numeric_columns = list(dfs.select_dtypes(exclude=[np.number]).columns)
print(non_numeric_columns)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
for col in non_numeric_columns:
    dfs[col] = le.fit_transform(dfs[col])

dfs.head(5)



# In[17]:

from causalnex.structure.notears import from_pandas
sm = from_pandas(dfs)


# In[18]:

sm


# In[19]:

from causalnex.plots import plot_structure
_, _, _ = plot_structure(sm)

sm.remove_edges_below_threshold(0.8)
_, _, _ = plot_structure(sm)




# In[21]:

sm = sm.get_largest_subgraph()

_, _, _ = plot_structure(sm)


# In[20]:

from causalnex.network import BayesianNetwork

bn = BayesianNetwork(sm)





# In[33]:
discretised_data = data.copy()

data_vals = {col: data[col].unique() for col in data.columns}

failures_map = {v: 'no-failure' if v == [0]
            else 'have-failure' for v in data_vals['failures']}

studytime_map = {v: 'short-studytime' if v in [1,2]
                 else 'long-studytime' for v in data_vals['studytime']}




# In[34]:

discretised_data["failures"] = discretised_data["failures"].map(failures_map)
discretised_data["studytime"] = discretised_data["studytime"].map(studytime_map)


# In[35]:

absences_map = {0: "No-absence", 1: "Low-absence", 2: "High-absence"}

G1_map = {0: "Fail", 1: "Pass"}
G2_map = {0: "Fail", 1: "Pass"}
G3_map = {0: "Fail", 1: "Pass"}

discretised_data["absences"] = discretised_data["absences"].map(absences_map)
discretised_data["G1"] = discretised_data["G1"].map(G1_map)
discretised_data["G2"] = discretised_data["G2"].map(G2_map)
discretised_data["G3"] = discretised_data["G3"].map(G3_map)


# In[36]:

# Split 90% train and 10% test
from sklearn.model_selection import train_test_split

train, test = train_test_split(discretised_data, train_size=0.9, test_size=0.1, random_state=7)



# In[37]:

"""

Model Probability
With the learnt structure model from earlier and the discretised data, we can now fit the probability distrbution of the Bayesian Network. The first step in this is specifying all of the states that each node can take. This can be done either from data, or providing a dictionary of node values. We use the full dataset here to avoid cases where states in our test set do not exist in the training set. For real-world applications, these states may need to be provided using the dictionary method.


"""


bn = bn.fit_node_states(discretised_data)




# In[38]:


# Fit Conditional Probability Distributions


bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")

bn.cpds["G1"]



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



