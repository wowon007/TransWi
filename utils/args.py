from argparse import ArgumentParser
from datasets import DS_NAMES
from methods import get_all_methods


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DS_NAMES,
                        help='Which dataset to perform experiments on.')

    parser.add_argument('--method', type=str, required=True,
                        help='Method name.', choices=get_all_methods())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')

    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')

    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')    

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')

    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging',default=False)
