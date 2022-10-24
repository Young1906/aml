from typing import List

class CateEncoder:
    def __init__(self, X:List[str]):
        X = list(set(X));
        
        self.idx2val = {i:v for (i, v) in enumerate(X)}
        self.val2idx = {v:i for (i, v) in enumerate(X)}

    def val(self, idx:int)->str:
        return self.idx2val[idx];

    def idx(self, val:str)->int:
        return self.val2idx[val]

    def __call__(self, val:str) -> int:
        return self.idx(val);
