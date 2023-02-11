import os
from typing import Any, Dict, Tuple, Union

import pandas as pd
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CommonLitDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: PreTrainedTokenizer,
                 labels: bool):
        self._inputs = tokenizer(df['excerpt'].values.tolist(),
                                 padding='max_length',
                                 truncation=True)
        self._labels = df['target'].values.tolist() if labels else None

    @staticmethod
    def create(data_dir: str,
               tokenizer: PreTrainedTokenizer,
               train: bool) -> Union[Dataset, Tuple[Dataset, Dataset]]:
        path = os.path.join(data_dir, f'{"train" if train else "test"}.csv')
        df = pd.read_csv(path)
        if train:
            train_df, val_df = train_test_split(df,
                                                test_size=0.2,
                                                random_state=0)
            return CommonLitDataset(train_df, tokenizer, train), \
                   CommonLitDataset(val_df, tokenizer, train)
        else:
            return CommonLitDataset(df, tokenizer, train)

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        item = {i: j[ind] for i, j in self._inputs.items()}
        if self._labels:
            item['label'] = self._labels[ind]
        return item

    def __len__(self) -> int:
        return len(self._labels)
