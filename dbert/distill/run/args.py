import argparse

from easydict import EasyDict as edict
import argconf


def read_args(default_config="confs/base.json", **parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--config", "-c", type=str, default=default_config)
    args, _ = parser.parse_known_args()
    options = argconf.options_from_json("confs/options.json")
    config = argconf.config_from_json(args.config)
    return edict(argconf.parse_args(options, config))