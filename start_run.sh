#!/bin/bash
# make sure container.simg is in your working directory
# make sure singularity is installed on your computer.

CODE_PATH="/mnt/hgfs/d/gitdev/_gran_dag"  # path to the code root
EXP_PATH="/mnt/hgfs/d/gitdev/_gran_dag/experiments"  # this folder will contain all artifacts saved by the program
DATA_PATH="/mnt/hgfs/d/gitdev/_gran_dag/data/data_p10_e10_n1000_GP"  # this should contain (data1.npy, DAG1.npy, CPDAG1.npy), (data2.npy, DAG2.npy, CPDAG2.npy), ...




MODEL="NonLinGaussANM"  # or NonLinGauss
DATA_INDEX=1  # Choose which dataset to use. Program will use (data${DATA_INDEX}.npy, DAG${DATA_INDEX}.npy, CPDAG${DATA_INDEX}.npy)
NUM_VARS=10  # should match the data provided




# Run python program. This is the Augmented Lagrangian procedure, without PNS. (add --pns to the command line to perform PNS)
# For GraN-DAG
singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX} --model $MODEL --train --to-dag --num-vars ${NUM_VARS} --jac-thresh"



# For cam
#singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python cam/main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX}"




# For dag_gnn
#singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python dag_gnn/main.py --exp-path /final_log/ --data-path /#dataset/ --i-dataset ${DATA_INDEX}"




# For notears
#singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python notears/main.py --exp-path /final_log/ --data-path /#dataset/ --i-dataset ${DATA_INDEX}"




# For random_baseline
#singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && #python random_baseline/main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX}"




# To use the baselines methods, replace main.py by the right path (e.g. for notears, it is notears/main.py)
# singularity exec --containall -B $DATA_PATH:/dataset/ -B $EXP_PATH:/final_log/ -B $CODE_PATH:/code/ ./container.simg bash -c "cd /code && python cam/main.py --exp-path /final_log/ --data-path /dataset/ --i-dataset ${DATA_INDEX}"



