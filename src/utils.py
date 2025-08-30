# Memory Cleanup Helper
import gc
import torch

def cleanup(model=None, optimizer=None):
    """
    Explicitly deletes model and optimizer and clears GPU cache.
    Helps to prevent CUDA out of memory errors in sequential experiments.
    """
    if optimizer:
        del optimizer
    if model:
        del model
    
    # Perform garbage collection
    gc.collect()
    
    # Empty the CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
