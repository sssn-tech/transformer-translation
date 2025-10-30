"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""

import torch
import os
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

class VocabTransform:
    def __init__(self, tokenize, init_token, eos_token, lower=True):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.lower = lower
        self.vocab = None

    def build_vocab(self, iterator, min_freq):
        def yield_tokens(data_iter):
            for src, tgt in data_iter:
                text = src if self.is_source else tgt
                yield self.preprocess(text)
        self.vocab = build_vocab_from_iterator(
            yield_tokens(iterator),
            specials=["<unk>", self.init_token, self.eos_token],
            min_freq=min_freq
        )
        self.vocab.set_default_index(self.vocab["<unk>"])

    def preprocess(self, text):
        if self.lower:
            text = text.lower()
        tokens = self.tokenize(text)
        return [self.init_token] + tokens + [self.eos_token]

    def numericalize(self, tokens):
        return torch.tensor(self.vocab(tokens), dtype=torch.long)

    def __call__(self, text):
        tokens = self.preprocess(text)
        return self.numericalize(tokens)


class DataLoader:
    source: VocabTransform = None
    target: VocabTransform = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('Dataset initialization started.')

    def make_dataset(self):
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
        
        def read_parallel_data(src_file, tgt_file):
            with open(src_file, 'r', encoding='utf-8') as sf, open(tgt_file, 'r', encoding='utf-8') as tf:
                src_lines = sf.readlines()
                tgt_lines = tf.readlines()
                
                if len(src_lines) != len(tgt_lines):
                    print(f"ERROR: Mismatched line count between source and target files! {src_file}: {len(src_lines)} lines, {tgt_file}: {len(tgt_lines)} lines")
                
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

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq)
        self.target.build_vocab(train_data, min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        def collate_fn(batch):
            src_batch = []
            tgt_batch = []
            for src, tgt in batch:
                src_tensor = self.source(src)
                tgt_tensor = self.target(tgt)
                src_batch.append(src_tensor)
                tgt_batch.append(tgt_tensor)
            src_batch = pad_sequence(src_batch, batch_first=True, padding_value=self.source.vocab["<unk>"])
            tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=self.target.vocab["<unk>"])
            return src_batch.to(device), tgt_batch.to(device)

        train_iterator = TorchDataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_iterator = TorchDataLoader(validate, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_iterator = TorchDataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print('Dataset initialization completed.')
        return train_iterator, valid_iterator, test_iterator