from typing import Any, TextIO, Tuple, Dict
import os

import torch


class Workspace(object):

    def __init__(self, folder: str, makedirs: bool = True) -> None:
        self.folder = folder
        if makedirs:
            os.makedirs(folder, exist_ok=True)

    def open(self, filename: str, *args, **kwargs) -> TextIO:
        return open(self.join(filename), *args, **kwargs)

    def torch_save(self, arg: Any, filename: str, **kwargs) -> None:
        torch.save(arg, self.join(filename), **kwargs)

    def torch_load(self, filename: str, **kwargs) -> Any:
        return torch.load(self.join(filename), **kwargs)

    def join(self, *filenames: Tuple[str]) -> str:
        return os.path.join(self.folder, *filenames)
