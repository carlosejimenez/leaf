import argparse

from .constants import DATASETS, SIM_TIMES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--init-path',
                        help='path to initialized model',
                        type=str,
                        default=None,
                        required=False)

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='small')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)
    # PR stands for personalization round - i.e. when to start personalization
    # Set this to -1 for no personalization ever
    parser.add_argument('-pr',
                        type=int,
                        default=0,
                        required=False)
    # Percent of online clients to use for creating personal models
    parser.add_argument('-personalization',
                        type=float,
                        default=.5,
                        required=False)
    parser.add_argument('-clusterer',
                        type=str,
                        choices=['Affinity', 'Agglo', 'Birch'],
                        default='Affinity')
    parser.add_argument('-num_clusters',
                        type=int,
                        default=10)

    return parser.parse_args()
