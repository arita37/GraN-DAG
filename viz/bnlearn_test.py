"""




conda install -c ankurankan pgmpy

#conda create -n env_BNLEARN python=3.6
#conda activate env_BNLEARN

conda install pytorch
pgmpy
numpy
pandas
tqdm
statsmodels
community
networkx==1.11
matplotlib==2.2.3

#conda install spyder




"""
####################################################################################################
####################################################################################################
import pandas as pd, numpy as np


def load(filename,  globs=None, varname=None) :
  import numpy as np, pandas as pd, pickle, os
  ### Name
  # print(filename)
  x = os.path.basename(filename)
  x = x if varname is None else varname
  x = x.split(".")[0].replace("-","_")

  if ".pkl" in filename :    obj = pickle.load( open(filename, mode="rb") , encoding="utf-8" )
  elif ".npy" in filename :  obj = np.load(filename)
  else  :                    obj = pd.read_csv(filename)

  if globs is not None :
    print(x, filename)
    globs[x] = obj
  else :
    return obj


def load_all(folder, globs, recursive=True) :
  import glob
  for f  in glob.glob(folder, recursive=recursive):
     try :
       load(f, globs=globs)
       # print(f)
     except Exception as e:
       print("error",  f, e)



#####################################
dir1 ="data/sachs/sachs/"


load_all( f"{dir1}/*.*", globs= globals() )



####### Details
load( f"{dir1}/categories.npy", globs= globals() )

load( f"{dir1}/idx2name.pkl", globs= globals() )

load( f"{dir1}/name2idx.pkl", globs= globals() )

load( f"{dir1}/sachs-header.npy", globs= globals() )





###############################################################
load_all( f"{dir1}/continuous/*.*", globs= globals() )



data1 = data1

CPDAG1

DAG1

a=1



a = DAG1




sachs_header



sachs-header


categories





import bnlearn as bn



# Example dataframe sprinkler_data.csv can be loaded with: 
df = bn.import_example()
# df = pd.read_csv('sprinkler_data.csv')

model = bn.structure_learning(df)
G = bn.plot(model)





import pgmpy


import numpy as np


a =1

