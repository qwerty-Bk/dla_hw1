import logging
import numpy as np
import torch
from typing import List
from speechbrain.utils.data_utils import pad_right_to

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    # TODO: your code here
    tensor_keys = ('audio', 'spectrogram', 'text_encoded')
    # for k, v in result_batch.items():
    #     if k not in tensor_keys:
    #         result_batch[k] = [v]

    max_tens = {k: 0 for k in tensor_keys}
    for d in dataset_items:
        for k in tensor_keys:
            # print(k, d[k].shape)
            if max_tens[k] == 0:
                max_tens[k] = list(d[k].shape)
            else:
                max_tens[k] = [max(x, y) for (x, y) in zip(max_tens[k], list(d[k].shape))]

    for i in range(len(dataset_items)):
        for k, v in dataset_items[i].items():
            if k in tensor_keys:
                value, prev_size = pad_right_to(v, max_tens[k])
                if i == 0:
                    result_batch[k] = value
                    result_batch[k + '_length'] = [v.shape[-1]]
                else:
                    result_batch[k] = torch.cat((result_batch[k], value))
                    result_batch[k + '_length'].append(v.shape[-1])
            else:
                if i == 0:
                    result_batch[k] = []
                result_batch[k].append(v)
    for k in tensor_keys:
        result_batch[k + '_length'] = torch.from_numpy(np.array(result_batch[k + '_length']))
    result_batch['spectrogram'] = torch.transpose(result_batch['spectrogram'], 1, 2)
    return result_batch
