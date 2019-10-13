from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable, Union, Tuple, Any
import enum


RawDictOption = Tuple[Union[Iterable[str], str], Dict[str, Any]]


class DictOption(object):

    def __init__(self, 
                 arg_names: Union[Iterable[str], str],
                 data: Dict[str, Any]) -> None:
        self.arg_names = arg_names
        self.data = data

    def default(self, default: Any):
        self.data['default'] = default
        return self

    def help(self, help_str: str):
        self.data['help'] = help_str
        return self

    def required(self, is_required: bool):
        self.data['required'] = is_required
        return self

    def __getitem__(self, idx):
        if idx == 0:
            return self.arg_names
        if idx == 1:
            return self.data
        raise IndexError


ArgDict = Iterable[Union[RawDictOption, DictOption]]


def add_dict_options(parser: ArgumentParser,
                     dict_options: ArgDict) -> None:
    for arg_names, dict_ in dict_options:
        if isinstance(arg_names, str):
            arg_names = (arg_names,)
        parser.add_argument(*arg_names, **dict_)


def opt(*args, **kwargs):
    return (args, kwargs)


def args_to_dict(args: Namespace, filter_: Iterable[str] = []):
    arg_dict = vars(args).copy()
    if filter_:
        keep_args = set(filter_)
        for k in list(arg_dict.keys()):
            if k not in keep_args:
                del arg_dict[k]
    return arg_dict


class OptionEnum(enum.Enum):
    DATA_DIR = DictOption('--data-dir',
                          dict(type=str,
                               required=True,
                               help='The input data dir. Should contain the task-specific data files.'))
    DO_TRAIN = DictOption('--do-train', 
                          dict(action='store_true',
                               help='Whether to run training.'))
    DO_EVAL = DictOption('--do-eval',
                         dict(action='store_true',
                              help='Whether to run eval on the dev set.'))
    DO_EVAL_ONLY = DictOption('--do-eval-only',
                              dict(action='store_true',
                                   help='Do evaluation only.'))
    LEARNING_RATE = DictOption('--learning-rate',
                               dict(default=5e-5,
                                    type=float,
                                    help='The initial learning rate.'))
    TRAIN_BATCH_SIZE = DictOption('--train-batch-size',
                                  dict(default=32,
                                       type=int,
                                       help='Total batch size for training.'))
    EVAL_BATCH_SIZE = DictOption('--eval-batch-size',
                                 dict(default=8,
                                      type=int,
                            help='Total batch size for eval.'))
    NUM_TRAIN_EPOCHS = DictOption('--num-train-epochs',
                                  dict(default=3,
                                       type=int,
                                       help='Total number of training epochs to perform.'))
    SEED = DictOption('--seed',
                      dict(type=int,
                           default=0,
                           help='Random seed for initialization.'))
    TASK_NAME = DictOption('--task-name', 
                           dict(type=str,
                                required=True,
                                help='The name of the task to train.'))
    HIDDEN_SIZE = DictOption('--hidden-size',
                             dict(type=int,
                                  default=150,
                                  help='The hidden size.'))
    RNN_TYPE = DictOption('--rnn-type',
                          dict(type=str,
                               default='lstm',
                               choices=['lstm', 'gru'],
                               help='The RNN type.'))
    OUTPUT_DIR = DictOption('--output-dir',
                            dict(type=str,
                                 required=True,
                                 help='The output directory where the model predictions and checkpoints will be written.'))
    TRAIN_FILE = DictOption('--train-file', dict(type=str, help='The training file.'))
    DEV_FILE = DictOption('--dev-file', dict(type=str, help='The dev file.'))
    TEST_FILE = DictOption('--test-file', dict(type=str, help='The test file.'))
    BERT_MODEL = DictOption('--bert-model', dict(type=str, required=True, help='Filepath of the BERT model.'))
    NO_BERT_TOKENIZE = DictOption('--no-bert-tokenize', dict(action='store_false', dest='do_bert_tokenize'))
    DO_LOWERCASE = DictOption('--do-lowercase', dict(action='store_true',
                                                     help='Set this flag if you are using an uncased model.'))
    WARMUP_PROPORTION = DictOption('--warmup-proportion', dict(default=0.1,
                                                               type=float,
                                                               help='Proportion of training to perform linear learning rate warmup for. '
                                                                    'E.g., 0.1 = 10%% of training.'))
    SPM_MODEL = DictOption('--spm-model', dict(type=str))