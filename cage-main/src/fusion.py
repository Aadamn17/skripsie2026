import torch

class TensorFuser:

    """A class to handle different fusion strategies for PyTorch tensors."""
    """Example usage: fuser = TensorFuser(tensors=[sensor_a_features, sensor_b_features])
    """
    
    def __init__(self, tensors: List[torch.Tensor]):
        """
        Initializes the fuser with a list of tensors.
        
        Args:
            tensors: A list of PyTorch tensors to fuse.
        """
        if not tensors:
            raise ValueError("The tensor list cannot be empty.")
        self.tensors = tensors

    def concatenate(self, dimension: int = 1) -> torch.Tensor:
        """
        Fuses tensors by joining them along an existing dimension.
        
        Args:
            dimension: The axis along which the tensors will be joined.
        """
        return torch.cat(self.tensors, dim=dimension)

    def stack(self, dimension: int = 0) -> torch.Tensor:
        """
        Fuses tensors by stacking them along a new dimension.
        
        Args:
            dimension: The index of the new dimension to create.
        """
        return torch.stack(self.tensors, dim=dimension)