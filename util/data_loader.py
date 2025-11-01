"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

import torch
import os
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from typing import Callable, Iterator


class VocabTransform:
    def __init__(self, tokenize: Callable[[str], list[str]], init_token: str, eos_token: str, lower: bool = True):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.lower = lower
        self.vocab: dict[str, int] | None = None
        self.itos: list[str] | None = None
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.unk_idx = 0
        self.pad_idx = 1

    def build_vocab(self, iterator: list[tuple[str, str]], min_freq: int):
        def yield_tokens(data_iter: list[tuple[str, str]]) -> Iterator[list[str]]:
            for src, tgt in data_iter:
                text = src if self.is_source else tgt
                yield self.preprocess(text)
        
        # Count token frequencies
        counter = Counter()
        for tokens in yield_tokens(iterator):
            counter.update(tokens)
        
        # Build vocabulary with all special tokens
        specials = [self.unk_token, self.pad_token, self.init_token, self.eos_token]
        self.itos = specials + [token for token, freq in counter.items() 
                                if freq >= min_freq and token not in specials]
        self.vocab = {token: idx for idx, token in enumerate(self.itos)}
        self.unk_idx = self.vocab[self.unk_token]
        self.pad_idx = self.vocab[self.pad_token]

    def preprocess(self, text: str) -> list[str]:
        if self.lower:
            text = text.lower()
        tokens = self.tokenize(text)
        return [self.init_token] + tokens + [self.eos_token]

    def numericalize(self, tokens: list[str]) -> torch.Tensor:
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]
        return torch.tensor(indices, dtype=torch.long)

    def __call__(self, text: str) -> torch.Tensor:
        tokens = self.preprocess(text)
        return self.numericalize(tokens)
    
    def __getitem__(self, token: str) -> int:
        """Allow vocab['<unk>'] style access"""
        return self.vocab.get(token, self.unk_idx)
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return len(self.vocab)


class DataLoader:
    source: VocabTransform | None = None
    target: VocabTransform | None = None

    def __init__(self, ext: tuple[str, str], tokenize_en: Callable, tokenize_de: Callable, 
                 init_token: str, eos_token: str):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('Dataset initialization started.')

    def make_dataset(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
        dataset_dir = os.path.join(os.getcwd(), 'dataset', 'multi30k')
        
        if self.ext == ('de', 'en'):
            src_lang, tgt_lang = 'de', 'en'
        elif self.ext == ('en', 'de'):
            src_lang, tgt_lang = 'en', 'de'
        else:
            raise ValueError(f"Invalid language extension: {self.ext}")
        
        train_src_path = os.path.join(dataset_dir, f'train.{src_lang}')
        train_tgt_path = os.path.join(dataset_dir, f'train.{tgt_lang}')
        val_src_path = os.path.join(dataset_dir, f'val.{src_lang}')
        val_tgt_path = os.path.join(dataset_dir, f'val.{tgt_lang}')
        test_src_path = os.path.join(dataset_dir, f'test.{src_lang}')
        test_tgt_path = os.path.join(dataset_dir, f'test.{tgt_lang}')
        
        def read_parallel_data(src_file: str, tgt_file: str) -> list[tuple[str, str]]:
            with open(src_file, 'r', encoding='utf-8') as sf, open(tgt_file, 'r', encoding='utf-8') as tf:
                src_lines = sf.readlines()
                tgt_lines = tf.readlines()
                
                if len(src_lines) != len(tgt_lines):
                    print(f"ERROR: Mismatched line count between source and target files! "
                          f"{src_file}: {len(src_lines)} lines, {tgt_file}: {len(tgt_lines)} lines")
                
                return [(src.strip(), tgt.strip()) for src, tgt in zip(src_lines, tgt_lines)]
        
        try:
            train_data = read_parallel_data(train_src_path, train_tgt_path)
            valid_data = read_parallel_data(val_src_path, val_tgt_path)
            test_data = read_parallel_data(test_src_path, test_tgt_path)
            
            print(f"{len(train_data)} training samples loaded.")
            print(f"{len(valid_data)} validation samples loaded.")
            print(f"{len(test_data)} test samples loaded.")
            
        except Exception as e:
            print(f"ERROR when loading dataset: {e}")
            print(f"Working directory: {os.getcwd()}")
            print(f"Dataset directory: {dataset_dir}")
            raise
        
        if self.ext == ('de', 'en'):
            self.source = VocabTransform(self.tokenize_de, self.init_token, self.eos_token)
            self.target = VocabTransform(self.tokenize_en, self.init_token, self.eos_token)
        elif self.ext == ('en', 'de'):
            self.source = VocabTransform(self.tokenize_en, self.init_token, self.eos_token)
            self.target = VocabTransform(self.tokenize_de, self.init_token, self.eos_token)
            
        self.source.is_source = True
        self.target.is_source = False

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        return self.train_data, self.valid_data, self.test_data

    def build_vocab(self, train_data: list[tuple[str, str]], min_freq: int):
        self.source.build_vocab(train_data, min_freq)
        self.target.build_vocab(train_data, min_freq)

    def make_iter(self, train: list[tuple[str, str]], validate: list[tuple[str, str]], 
                  test: list[tuple[str, str]], batch_size: int, device: torch.device):
        def collate_fn(batch: list[tuple[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
            src_batch = []
            tgt_batch = []
            for src, tgt in batch:
                src_tensor = self.source(src)
                tgt_tensor = self.target(tgt)
                src_batch.append(src_tensor)
                tgt_batch.append(tgt_tensor)
            src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.source.pad_idx)
            tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.target.pad_idx)
            return src_batch.to(device), tgt_batch.to(device)

        train_iterator = TorchDataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_iterator = TorchDataLoader(validate, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_iterator = TorchDataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print('Dataset initialization completed.')
        return train_iterator, valid_iterator, test_iterator