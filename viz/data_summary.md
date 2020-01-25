




3 types of data :


  DAG.npy   : DAG as matrix

  data.npy

  CPDAG.npy


  Complete partial Dependence DAG :
     




        # Load the graph
        adjacency = np.load(os.path.join(file_path, "DAG{}.npy".format(i_dataset)))


        # Load data
        self.data_path = os.path.join(file_path, "data{}.npy".format(i_dataset))




        # Shuffle and filter examples
        shuffle_idx = np.arange(data.shape[0])
        self.random.shuffle(shuffle_idx)
        data = data[shuffle_idx[: train_samples + test_samples]]




def sample(self, batch_size):
        sample_idxs = self.random.choice(np.arange(int(self.num_samples)), size=(int(batch_size),), replace=False)
        samples     = self.dataset[torch.as_tensor(sample_idxs).long()]
        return samples, torch.ones_like(samples)  # second output is mask (for intervention in the future)







## sachs protein

  # Imports datasets used in the paper: 
  #Sachs, Karen, et al. "Causal protein-signaling networks derived from multiparameter single-cell data." Science 308.5721 (2005): 523-529.
  # Imports the data from the Data Files directory downloaded as a zip from the publication's site on Science.com
  # Direct link to the file as of Dec 15, 2014: 
  # http://www.sciencemag.org/content/suppl/2005/04/21/308.5721.523.DC1/Sachs.SOM.Datasets.zip
  #





############# 4.1 Synthetic data  ##############################################################
We have generated different data set types which vary along three dimensions: 
  
   number of nodes,
   level of edge sparsity and 
   graph type. 


For each data set type, we have sampled 10 data sets of size
n = 1000. We consider two different types of graphs:
     Erd ˝os–Rényi (ER) and 
     scale-free (SF) graphs.


Both types differ in the way graphs are randomly generated (see Appendix A.5).
Given a data set type, a data set is sampled as follows: First, a ground truth DAG G is randomly sampled following either the ER or the SF scheme. Then, the data is generated following


Xi|Xπ ∼ N (fi(Xπ), sigma**2)


∀i with the functions fi
independently sampled from a Gaussian process with bandwidth one and σ2i
sampled uniformly. This setup is especially interesting to consider
since, as mentioned in Section 2.2, we know the DAG G to be identifiable from the distribution [27].
This ensures that finding the correct DAG via maximum likelihood is not impossible.

In those experiments, each NN learned by GraN-DAG outputs a Gaussian mean µˆ(i)

Note that the linear method NOTEARS and the nonlinear methods CAM and RESIT all
make the correct Gaussian model assumption.


In this section, we compare GraN-DAG to various baselines (both in the continuous and combinatorial
paradigm), 
    namely DAG-GNN [35], 
           NOTEARS [37], 
           RESIT [27] and 
           CAM [5]. 

Those methods are discussed in Section 5. As a sanity check, we report the performance of random graphs sampled
using the Erd ˝os–Rényi (ER) scheme described in Appendix A.5 (denoted by RANDOM). 

For each approach, we evaluate the estimated graph on two metrics: 

  the structural hamming distance (SHD)
  the structural interventional distance (SID) [25]. 


The former simply counts the number of missing, falsely detected or reversed edges. 
The latter is especially well suited for causal inference
since it counts the number of couples (i, j) such that the interventional distribution 
p(xi|do(Xj = ¯x))
would be miscalculated if we were to use the estimated graph to form the parent adjustement set. See

Appendix A.7 for more details on SHD and SID. We consider both synthetic and real-world data sets.
Since the performance of GES [6] and PC [32] are almost never on par





###### Human Cells Intervention :

This data set contains both
observational and interventional data [18, 28].



[18] D. Koller and N. Friedman. Probabilistic Graphical Models: Principles and Techniques -
Adaptive Computation and Machine Learning. MIT Press, 2009.


[28] J. Peters, D. Janzing, and B. Schölkopf. Elements of Causal Inference - Foundations and
Learning Algorithms. MIT Press, 2017.


































