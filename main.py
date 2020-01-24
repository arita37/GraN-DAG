# coding=utf-8
"""
GraN-DAG

Copyright © 2019 Authors of Gradient-Based Neural DAG Learning



"""
import os
import argparse
import copy
import numpy as np

from gran_dag.main import main


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    def add(*w, **kw):
       p.add_argument(*w, **kw)

    # experiment
    add('--exp-path', type=str, default='/exp', help='Path to experiments')
    add('--pns', action="store_true",help='Run `pns` function, get /pns folder')
    add('--train', action="store_true", help='Run `train` function, get /train folder')
    add('--to-dag', action="store_true",  help='Run `to_dag` function, get /to-dag folder')
    add('--cam-pruning', action="store_true",  help='Run `cam_pruning` function, get /cam-pruning folder')
    add('--retrain', action="store_true",  help='after to-dag or pruning, retrain model from scratch before reporting nll-val')
    add('--random-seed', type=int, default=42, help="Random seed for pytorch and numpy")

    # data
    add('--data-path', type=str, default=None,   help='Path to data files')
    add('--i-dataset', type=str, default=None,  help='dataset index')
    add('--num-vars', type=int, default=2,  help='Number of variables')
    add('--train-samples', type=int, default=0.8,  help='Number of samples used for training (default is 80% of the total size)')
    add('--test-samples', type=int, default=None, help='Number of samples used for testing (default is whatever is not used for training)')
    add('--train-batch-size', type=int, default=64,  help='number of samples in a minibatch')
    add('--num-train-iter', type=int, default=100000, help='number of meta gradient steps')
    add('--normalize-data', action="store_true",  help='(x - mu) / std')

    # model
    add('--model', type=str, required=True,    help='model class')
    add('--num-layers', type=int, default=2,   help="number of hidden layers")
    add('--hid-dim', type=int, default=10,    help="number of hidden units per layer")
    add('--nonlin', type=str, default='leaky-relu',   help="leaky-relu | sigmoid")

    # optimization
    add('--optimizer', type=str, default="rmsprop",   help='sgd|rmsprop')
    add('--lr', type=float, default=1e-3,     help='learning rate for optim')
    add('--lr-reinit', type=float, default=None,  help='Learning rate for optim after first subproblem. Default mode reuses --lr.')
    add('--scale-lr-with-mu', action="store_true",   help='Scale the learning rate wrt mu in the augmented lagrangian.')
    add('--stop-crit-win', type=int, default=100,   help='window size to compute stopping criterion')

    # pns, pruning and thresholding
    add('--pns-thresh', type=float, default=0.75,    help='threshold in PNS')
    add('--num-neighbors', type=int, default=None,    help='number of neighbors to select in PNS')
    add('--edge-clamp-range', type=float, default=1e-4,    help='as we train, clamping the edges (i,j) to zero when prod_ij is that close to zero. '
                             '0 means no clamping. Uses masks on inputs. Once an edge is clamped, no way back.')
    add('--cam-pruning-cutoff', nargs='+',    default=np.logspace(-6, 0, 10),     help='list of cutoff values. Higher means more edges')

    # Augmented Lagrangian options
    add('--omega-lambda', type=float, default=1e-4,     help='Precision to declare convergence of subproblems')
    add('--omega-mu', type=float, default=0.9,     help='After subproblem solved, h should have reduced by this ratio')
    add('--mu-init', type=float, default=1e-3,      help='initial value of mu')
    add('--lambda-init', type=float, default=0.,      help='initial value of lambda')
    add('--h-threshold', type=float, default=1e-8,       help='Stop when |h|<X. Zero means stop AL procedure only when h==0. Should use --to-dag even '
                             'with --h-threshold 0 since we might have h==0 while having cycles (due to numerical issues).')

    # misc
    add('--norm-prod', type=str, default="paths",    help='how to normalize prod: paths|none')
    add('--square-prod', action="store_true",    help="square weights instead of absolute value in prod")
    add('--jac-thresh', action="store_true",      help='threshold using the Jacobian instead of prod')
    add('--patience', type=int, default=10,      help='Early stopping patience in --retrain.')

    # logging
    add('--plot-freq', type=int, default=10000,    help='plotting frequency')
    add('--no-w-adjs-log', action="store_true",    help='do not log weighted adjacency (to save RAM). One plot will be missing (A_\phi plot)')

    # device and numerical precision
    add('--gpu', action="store_true",      help="Use GPU")
    add('--float', action="store_true",      help="Use Float precision")

    main(p.parse_args())

