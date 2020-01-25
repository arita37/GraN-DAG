import bnlearn as bnlearn
import numpy as np
import pandas as pd

cpdag = pd.DataFrame(np.load("CPDAG1.npy"))
dag = pd.DataFrame(np.load("DAG1.npy"))
data = pd.DataFrame(np.load("data1.npy"))


model = bnlearn.structure_learning(dag)
G = bnlearn.plot(model)

#model = bnlearn.import_DAG(CPD=False)
#model_update = bnlearn.parameter_learning(model, dag)
# = bnlearn.plot(model)

