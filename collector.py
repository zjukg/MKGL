import numpy as np
import pandas as pd
import torch
import easydict
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class MKGLDataCollector:
    dataset: object = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        first = features[0]
        batch_size = len(features)

        batch = {}
        for k, v in first.items():
            batch[k] = [f[k] for f in features]

        batch['input_text'] = batch['input_text']+batch['inv_input_text']
        batch.update(self.dataset.tokenizer(batch['input_text'], padding=True))
        batch['input_length'] = np.sum(batch['attention_mask'], axis=1)

        split = batch['split'][0]
        del batch['input_text'], batch['inv_input_text'], batch['split']
        for k, v in batch.items():
            batch[k] = torch.tensor(batch[k])
        batch['split'] = split

        if batch['split'] != 'train':
            return {'batch': easydict.EasyDict(batch), 'label': torch.ones(batch_size, dtype=torch.bfloat16)}
        else:
            return {'batch': easydict.EasyDict(batch)}
