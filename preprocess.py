import dataset
from typing import Union
import numpy as np
import swifter
import pandas as pd
from datasets import Dataset
from torchdrug import core
from torchdrug.utils import pretty
from transformers import AutoTokenizer
import argparse, easydict, yaml
import json
import pickle
import os.path as osp
import os




class Prompter(object):

    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


class InductiveKGCDataset(object):

    def __init__(self, args, kgdata, tokenizer):
        self.args = args
        self.kgdata = kgdata
        self.tokenizer = tokenizer
        self.prompter = Prompter('alpaca_short', verbose=False)
        self.inv_prefix = '/inv'
        self.inv_fine_prefix = 'inverse of '

        self.read_vocab()
        self.read_data()
        self.add_input_text()
        self.post_process()

        self.saved_dir = 'data/preprocessed/'
        self.save()

    def read_vocab(self):
        kgdata = self.kgdata

        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        if 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'

        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values,
                             index=ent_name['raw_name'].values)
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values,
                             index=rel_name['raw_name'].values)

        trans_ent_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.transductive_vocab)), 'raw_name': kgdata.transductive_vocab, 'transductive': 1}, )
        ind_ent_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.inductive_vocab)), 'raw_name': kgdata.inductive_vocab, 'transductive': 0}, )
        ent_vocab_df = pd.concat(
            [trans_ent_vocab_df, ind_ent_vocab_df], ignore_index=True)

        ent_vocab_df['fine_name'] = ent2text[ent_vocab_df.raw_name.values].values

        rel_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.relation_vocab)), 'raw_name': kgdata.relation_vocab, 'transductive': 0})
        rel_vocab_df['fine_name'] = rel2text[rel_vocab_df.raw_name.values].values

        inv_rel_vocab_df = rel_vocab_df.iloc[:]
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + \
            inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + \
            inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat(
            [rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby(
            'fine_name').apply(process_overlapped_name)
        ent_vocab_df = ent_vocab_df.droplevel('fine_name').sort_index()

        rel_vocab_df = rel_vocab_df.groupby(
            'fine_name').apply(process_overlapped_name)
        rel_vocab_df = rel_vocab_df.droplevel('fine_name').sort_index()

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        def tokenize(vocab_df):
            tokenizer.add_tokens(vocab_df['token_name'].values.tolist())
            vocab_df['token_index'] = [tokenizer.get_added_vocab()[tn]
                                       for tn in vocab_df['token_name'].values]

            # to avoid some entities having identical name
            raw_names, indices = np.unique(
                vocab_df['raw_name'].values, return_index=True)
            rawname2tokenid = pd.Series(
                vocab_df['token_index'].values[indices], index=raw_names)

            vocab_df.set_index('token_index', inplace=True)

            fine_name = [str(n).strip() for n in vocab_df['fine_name'].values]

            token_ids = tokenizer(
                fine_name, add_special_tokens=False, truncation=True, padding=True).input_ids

            vocab_df['text_token_ids'] = token_ids

            return vocab_df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize(vocab_df)

    def read_data(self):
        kgdata = self.kgdata
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)

            df = pd.DataFrame(subset[:], columns=['h_id', 't_id', 'r_id'])
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            df['inv_r_tokenid'] = self.rawname2tokenid[self.inv_prefix +
                                                       df['r_raw'].values].values

            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values,
                                                 'fine_name'].values

            return df

        train_df = convert_to_df(
            train_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        valid_df = convert_to_df(
            valid_set, kgdata.transductive_vocab, kgdata.relation_vocab)
        test_df = convert_to_df(test_set, kgdata.inductive_vocab,
                                kgdata.relation_vocab)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def add_input_text(self):
        print('##########Add input text##########')

        train_df, valid_df, test_df = self.train_df, self.valid_df, self.test_df
        vocab_df = self.vocab_df

        def produce_input_text(row):

            h_info = vocab_df.loc[row['h_tokenid']]
            t_info = vocab_df.loc[row['t_tokenid']]
            r_info = vocab_df.loc[row['r_tokenid']]
            inv_r_info = vocab_df.loc[row['inv_r_tokenid']]

            h = h_info['token_name']
            t = t_info['token_name']
            r = r_info['token_name']
            inv_r = inv_r_info['token_name']

            h_des = h_info['fine_name']
            t_des = t_info['fine_name']
            r_des = r_info['fine_name']
            inv_r_des = inv_r_info['fine_name']

            instruction = f'Suppose that you are an excellent linguist studying a three-word language. Given the following dictionary:\n\n Input\tType\tDescription\n{h}\tHead entity\t{h_des}\n{r}\tRelation\t{r_des}\n\nPlease complete the last word (?) of the sentence: {h}{r}?'
            inv_instruction = f'Suppose that you are an excellent linguist studying a three-word language. Given the following dictionary:\n\n Input\tType\tDescription\n{t}\tHead entity\t{t_des}\n{inv_r}\tRelation\t{inv_r_des}\n\nPlease complete the last word (?) of the sentence: {t}{inv_r}?'

            row['input_text'] = self.prompter.generate_prompt(instruction, label=f'{h}{r}')
            row['inv_input_text'] = self.prompter.generate_prompt(
                inv_instruction, label=f'{t}{inv_r}')

            return row

        test_df = test_df.swifter.apply(produce_input_text, axis=1)
        valid_df = valid_df.swifter.apply(produce_input_text, axis=1)
        train_df = train_df.swifter.apply(produce_input_text, axis=1)

        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df

    def _to_hf_dataset(self, df):

        return Dataset.from_pandas(df)

    def post_process(self):
        print('##########Post process: convert to hf datasets##########')

        self.train_data = self._to_hf_dataset(self.train_df)
        self.valid_data = self._to_hf_dataset(self.valid_df)
        self.test_data = self._to_hf_dataset(self.test_df)

    def save(self):
        saved_dir = self.saved_dir
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir)

        file_path = saved_dir+args.config_name+'.pkl'
        print('##########Save dataset in %s############' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        print('##########Load dataset from %s############' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class KGCDataset(InductiveKGCDataset):

    def read_vocab(self):
        kgdata = self.kgdata

        if 'fb15' in self.args.config_name:
            name_prefix = './data/names/fb15k237/'
        if 'wn18' in self.args.config_name:
            name_prefix = './data/names/wn18rr/'

        ent_name = pd.read_csv(name_prefix+'entity.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'], dtype=str)
        ent2text = pd.Series(ent_name['fine_name'].values,
                             index=ent_name['raw_name'].values)
        rel_name = pd.read_csv(name_prefix+'relation.txt',
                               sep='\t', header=None, names=['raw_name', 'fine_name'])
        rel2text = pd.Series(rel_name['fine_name'].values,
                             index=rel_name['raw_name'].values)

        ent_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.entity_vocab)), 'raw_name': kgdata.entity_vocab, 'transductive': 1}, )
        ent_vocab_df['fine_name'] = ent2text[ent_vocab_df.raw_name.values].values

        rel_vocab_df = pd.DataFrame({'kg_id': range(
            len(kgdata.relation_vocab)), 'raw_name': kgdata.relation_vocab, 'transductive': 0})
        rel_vocab_df['fine_name'] = rel2text[rel_vocab_df.raw_name.values].values

        inv_rel_vocab_df = rel_vocab_df.iloc[:]
        inv_rel_vocab_df['kg_id'] += len(inv_rel_vocab_df)
        inv_rel_vocab_df['raw_name'] = self.inv_prefix + \
            inv_rel_vocab_df['raw_name']
        inv_rel_vocab_df['fine_name'] = self.inv_fine_prefix + \
            inv_rel_vocab_df['fine_name']

        rel_vocab_df = pd.concat(
            [rel_vocab_df, inv_rel_vocab_df], ignore_index=True)

        def process_overlapped_name(rows):
            if len(rows) > 1:
                rows.loc[:, 'fine_name'] = rows.loc[:, 'fine_name'] + \
                    [' #%i' % i for i in range(1, len(rows)+1)]
            return rows

        ent_vocab_df = ent_vocab_df.groupby(
            'fine_name').apply(process_overlapped_name)
        ent_vocab_df = ent_vocab_df.droplevel('fine_name').sort_index()

        rel_vocab_df = rel_vocab_df.groupby(
            'fine_name').apply(process_overlapped_name)
        rel_vocab_df = rel_vocab_df.droplevel('fine_name').sort_index()

        ent_vocab_df['entity'] = 1
        rel_vocab_df['entity'] = 0
        vocab_df = pd.concat([ent_vocab_df, rel_vocab_df], ignore_index=True)
        vocab_df['token_name'] = '<rdf: ' + vocab_df['fine_name'] + '>'

        def tokenize(vocab_df):
            tokenizer.add_tokens(vocab_df['token_name'].values.tolist())
            vocab_df['token_index'] = [tokenizer.get_added_vocab()[tn]
                                       for tn in vocab_df['token_name'].values]

            rawname2tokenid = pd.Series(
                vocab_df['token_index'].values, index=vocab_df['raw_name'].values)

            vocab_df.set_index('token_index', inplace=True)

            fine_name = [str(n).strip() for n in vocab_df['fine_name'].values]

            token_ids = tokenizer(
                fine_name, add_special_tokens=False, truncation=True, padding=True).input_ids

            vocab_df['text_token_ids'] = token_ids

            return vocab_df, rawname2tokenid

        self.vocab_df, self.rawname2tokenid = tokenize(vocab_df)

    def read_data(self):
        kgdata = self.kgdata
        train_set, valid_set, test_set = kgdata.split()

        def convert_to_df(subset, ent_vocab, rel_vocab):
            ev = pd.Series(ent_vocab)
            rv = pd.Series(rel_vocab)

            df = pd.DataFrame(subset[:], columns=['h_id', 't_id', 'r_id'])
            df['h_raw'] = ev[df['h_id'].values].values
            df['t_raw'] = ev[df['t_id'].values].values
            df['r_raw'] = rv[df['r_id'].values].values

            df['h_tokenid'] = self.rawname2tokenid[df['h_raw'].values].values
            df['t_tokenid'] = self.rawname2tokenid[df['t_raw'].values].values
            df['r_tokenid'] = self.rawname2tokenid[df['r_raw'].values].values
            df['inv_r_tokenid'] = self.rawname2tokenid[self.inv_prefix +
                                                       df['r_raw'].values].values

            df['h_fine'] = self.vocab_df.loc[df['h_tokenid'].values, 'fine_name'].values
            df['t_fine'] = self.vocab_df.loc[df['t_tokenid'].values, 'fine_name'].values
            df['r_fine'] = self.vocab_df.loc[df['r_tokenid'].values, 'fine_name'].values
            df['inv_r_fine'] = self.vocab_df.loc[df['inv_r_tokenid'].values,
                                                 'fine_name'].values

            return df

        train_df = convert_to_df(
            train_set, kgdata.entity_vocab, kgdata.relation_vocab)
        valid_df = convert_to_df(
            valid_set, kgdata.entity_vocab, kgdata.relation_vocab)
        test_df = convert_to_df(test_set, kgdata.entity_vocab,
                                kgdata.relation_vocab)

        train_df['split'] = 'train'
        valid_df['split'] = 'valid'
        test_df['split'] = 'test'
        self.train_df, self.valid_df, self.test_df = train_df, valid_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument("--config", "-c", type=str,
                        default='config/fb15k237.yaml')
    parser.add_argument("--version", "-v", type=str,
                        default='')
    parser.add_argument("--seed", "-s", type=str,
                        default=42)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = easydict.EasyDict(yaml.safe_load(f))
        if 'ind' in args.config:
            assert args.version
            cfg.dataset.version = args.version

    config_name = args.config.split('/')[-1].split('.')[0]
    if hasattr(cfg.dataset, 'version'):
        config_name += '_' + cfg.dataset.version
    args.config_name = config_name

    print('***************Read dataset from A*Net***************')
    print("Config file: %s" % args.config)
    print("Config name: %s" % args.config_name)
    print(pretty.format(cfg))
    kgdata = core.Configurable.load_config_dict(cfg.dataset)


    print('***************Load tokenizer***************')
    tokenizer = AutoTokenizer.from_pretrained(**cfg.tokenizer)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'right'
    if 'ind' in args.config:
        dataset = InductiveKGCDataset(args, kgdata, tokenizer)
    else:
        dataset = KGCDataset(args, kgdata, tokenizer)

