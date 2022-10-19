import torch, os, glob
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

class Triplets(torch.utils.data.Dataset):
    def __init__(path:str):
        """
        ================================================================================
        Desc:
        --------------------------------------------------------------------------------
        
            Set of triplets (A, P, N), in which:
                - A : Anchor, an image of any class
                - P : Positive, another sample with same class with A
                - N : negative, sample with different class to A

        ================================================================================
        Args
        --------------------------------------------------------------------------------
            - path : str 
                path to folder containing training image with following structure
                
                ```
                root/
                    class_1/
                        sample_1
                        sample_2
                        ...
                    class_2
                        sample_n
                        ...
                ```
        """

        super().__init__();


        # path to img folder
        self.pth = path;

        #
    


    def 
