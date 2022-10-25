import torch.utils.data
from torch.utils.data import DataLoader
from sys import float_info
import numpy as np
import itertools, operator


class InRAMDataset(torch.utils.data.Dataset):
    """
    Parent class for storing and shuffling data.
    """

    def __init__(self, configs):
        """
        Attributes
        ----------
        configs: list of configuration objects
        """

        self.configs = configs
        #for i in range(0,196):
        #    print(configs[i].inputs)
        #    print(configs[i].targets)

        #assert(False)

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        pass

class InRAMDatasetPyTorch(InRAMDataset):
    """
    
    """

    def __getitem__(self, idx):
        """
        Convert configuration quantities to tensors and return them, for a single configuration in a 
        batch. 
        TODO: We could eliminate this costly conversion by storing all of these as tensors instead 
        of numpy arrays from the beginning, when processing configs in the Calculator class.
        """

        inputs = torch.unsqueeze(torch.tensor(self.configs[idx].inputs).float(),dim=0)
        targets = torch.unsqueeze(torch.tensor(self.configs[idx].targets).float(), dim=0)
        #inputs = torch.tensor(self.configs[idx].inputs).float()
        #targets = torch.tensor(self.configs[idx].targets).float()

        configuration = {'inputs': inputs,
                         'targets': targets}

        return configuration


def torch_collate(batch):
    """
    Collate batch of data, which collates a stack of configurations from Dataset into a batch.
    """

    batch_of_inputs = torch.cat([conf['inputs'] for conf in batch], dim=0)
    batch_of_targets = torch.cat([conf['targets'] for conf in batch], dim=0)

    #print(batch_of_inputs)
    #print(batch_of_targets)

    #print("-------")
    #assert(False)

    collated_batch = {'inputs': batch_of_inputs,
                      'targets': batch_of_targets}

    return collated_batch