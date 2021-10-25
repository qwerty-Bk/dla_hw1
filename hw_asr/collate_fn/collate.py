import logging
import numpy as np
import torch
from typing import List
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    tensor_keys = ('audio', 'spectrogram', 'text_encoded')

    for i in range(len(dataset_items)):
        for k, v in dataset_items[i].items():
            if k in tensor_keys:
                v = v.squeeze(0)
                if k == 'spectrogram':
                    v = torch.transpose(v, 0, 1)
                if i == 0:
                    result_batch[k] = []
                    result_batch[k + '_length'] = []
                result_batch[k].append(v)
                result_batch[k + '_length'].append(v.shape[0])
            else:
                if i == 0:
                    result_batch[k] = []
                result_batch[k].append(v)
    for k in tensor_keys:
        result_batch[k] = pad_sequence(result_batch[k], batch_first=True, padding_value=-1)
        result_batch[k + '_length'] = torch.from_numpy(np.array(result_batch[k + '_length']))
    return result_batch
