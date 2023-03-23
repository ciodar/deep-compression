from torch.nn.utils.prune import l1_unstructured, ln_structured, global_unstructured, identity
from compression.pruning import l1_threshold, get_pruned, ThresholdPruning
from compression.quantization import is_quantized, density_quantization, forgy_quantization, linear_quantization
