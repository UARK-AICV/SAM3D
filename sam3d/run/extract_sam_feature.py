import argparse

from sam3d.run.default_configuration import get_default_configuration
from sam3d.paths import default_plans_identifier
from sam3d.utilities.task_name_id_conversion import convert_id_to_task_name
import numpy as np
import time 
import torch

seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# There are many implementations of cudnn algorithms, if you want to allow only deterministic algos for reproducibility, then 
# set 'deterministic = True'. However, some operations (pooling, for example) which is nondeterministic, will throw errors.
torch.backends.cudnn.deterministic = False
# If the input size does not vary each iteration and the model remains the same, then set 'torch.backends.cudnn.benchmark = True'
# in order to use the appreciate algorithm for your hardware. Otherwise, you should turn it off.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

