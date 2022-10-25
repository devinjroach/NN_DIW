import numpy as np

class Configuration():
    """
    Class to store training data for each config, allows us to easily collate and process configs in 
    the dataloader.
    """
    def __init__(self,inputs,targets):
        self.inputs = inputs
        self.targets = targets