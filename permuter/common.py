from typing import Sequence

import torch


class PermDict:
    def __init__(self, keys:Sequence[str]) -> None:
        """"""
        self._dict:dict = {
            key: None
            for key in keys
        }
    
    def _check_key(self, key:str):
        if key not in self._dict.keys():
            raise KeyError(f"Key: {key} not a valid key in Permutation dictionary")
    
    @classmethod
    def from_dict(cls, perm_dict:dict[str, torch.Tensor]):
        tmp = cls(list(perm_dict.keys()))
        for key in perm_dict.keys():
            tmp[key] = perm_dict[key]
        return tmp
    
    def __setitem__(self, key:str, item:torch.Tensor):
        self._check_key(key)
        self._dict.update({key: item})
    
    def __getitem__(self, key:str):
        self._check_key(key)
        return self._dict[key]

    def __call__(self) -> dict:
        return self._dict
    