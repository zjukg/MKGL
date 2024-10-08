import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data as torch_data

from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

from torchdrug import core, tasks
from gnn.model import PNA
from retriever import *

class MKGLConfig(LlamaConfig):
    model_type = 'mkgl_config'

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

class MKGL(LlamaForCausalLM):
    config_class = MKGLConfig

    def __init__(self, config):
        super().__init__(config)

    def init_kg_specs(self, kgl2token, orig_vocab_size, cfg,):
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        device = self.lm_head.weight.device
        self.context_retriever = ContextRetriever(cfg.context_retriever, self.get_input_embeddings().weight.data, kgl2token, orig_vocab_size).to(device)
        self.score_retriever = ScoreRetriever(cfg.score_retriever, self.lm_head.weight.data, kgl2token, orig_vocab_size).to(device)

        # self._init_kg_score(len(kgl_vocab), r)

    def _init_kg_score(self, num_kg_tokens, ent_inter_emb_dim=64):
        device = self.lm_head.weight.device

        def kg_lora_layer(output_dim=num_kg_tokens):
            linear_a = nn.Linear(
                self.config.hidden_size, ent_inter_emb_dim, bias=False, dtype=torch.float, device=device)
            linear_b = nn.Linear(
                ent_inter_emb_dim, output_dim, bias=False, dtype=torch.float, device=device)

            nn.init.xavier_normal_(linear_a.weight)
            # nn.init.xavier_normal_(linear_b.weight)
            nn.init.zeros_(linear_b.weight)
            return nn.Sequential(OrderedDict([
                ('linear_a', linear_a),
                ('dropout', nn.Dropout(.2)),
                ('linear_b', linear_b),
            ]))

        self.kg_score = kg_lora_layer()


    def forward(
        self,
        h_id,
        r_id,
        t_id,
        h_kgl_tokenid,
        r_kgl_tokenid,
        graph,
        all_index,
        all_kgl_index,
        input_ids,
        attention_mask,
        input_length,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size = h_kgl_tokenid.shape[0]
        device = self.lm_head.weight.device

        mask = input_ids < self.orig_vocab_size
        token_embs = self.get_input_embeddings()(input_ids[mask])
        kgl_token_embs = self.context_retriever(input_ids[~mask], graph, all_index, all_kgl_index)

        rel_token_embs = self.context_retriever(r_kgl_tokenid, graph, all_index, all_kgl_index)

        input_embs = torch.zeros(
            *input_ids.shape, self.config.hidden_size, dtype=torch.half).to(device)
        input_embs[mask] = token_embs.type(input_embs.dtype)
        input_embs[~mask] = kgl_token_embs.type(input_embs.dtype)

        transformer_outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=input_embs,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # batch_size, seq_len, hidden_state
        hidden_states = transformer_outputs[0]

        # select the last output of llm, batch_size x hidden_size
        hr_hidden_states = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), input_length-1]

        rel_hidden_states = hidden_states[torch.arange(
            batch_size, device=hidden_states.device), input_length-2]


        pred = self.score_retriever(h_id, r_id, t_id, hr_hidden_states, rel_token_embs, graph, all_index, all_kgl_index)
        return pred
    

    def get_input_kg_embeddings(self, kgl_token_ids):
        kgl_token_ids = kgl_token_ids - self.orig_vocab_size
        if token_embs is None:
            token_embs = self.get_input_embeddings().weight.data
        device = token_embs.device

        kg_token_ids = self.kgl_vocab
        kg_token_mask = kg_token_ids > 0
        kg_token_lengths = kg_token_mask.float().sum(axis=-1)

        # shape: num_ents x hidden_size
        results = (token_embs[kg_token_ids.to(device)] *
                   kg_token_mask.unsqueeze(-1).to(device)).sum(axis=1).squeeze() / kg_token_lengths.unsqueeze(-1).float().to(device)

        if self.apply_norm:
            results = self.norm(results)
        return results
    
    def norm(self, x):
        return F.normalize(x, p=2, dim=1)

class KGL4KGC(nn.Module):

    def __init__(self, config, llmodel, dataset):
        super().__init__()
        self.llmodel = llmodel
        self.dataset = dataset
        self.num_negative = config.num_negative
        self.adversarial_temperature = config.adversarial_temperature
        self.strict_negative = config.strict_negative
        
        train_set, valid_set, test_set = dataset.kgdata.split()
        self.preprocess(train_set, valid_set, test_set)

    @property
    def device(self):
        return self.llmodel.lm_head.weight.device

    
    def loss(self, pred, target, all_loss=None):
        metric = {}
        target = torch.zeros_like(pred)
        target[:, 0] = 1
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none")

        neg_weight = torch.ones_like(pred)
        if self.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    pred[:, 1:] / self.adversarial_temperature, dim=-1)
        else:
            neg_weight[:, 1:] = 1 / self.num_negative
        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        
        if all_loss is not None:
            loss = loss + all_loss
            
        metric['loss'] = loss
        
        return loss, metric
    
    def forward(self, batch, all_loss=None, metric=None, label=None):
        device = batch.h_id.device
        
        if self.training:
            all_loss = torch.tensor(0, dtype=torch.float, device=device)
            pred = self.predict(batch, all_loss, metric)
            
            target = torch.zeros_like(pred)
            target[:, 0] = 1
            
            return self.loss(pred, target)
        
        else:
            with torch.no_grad():
                pred, (mask, target) = self.predict_and_target(batch)
                label = torch.zeros_like(pred)
                label[:, target] = 1
                loss, _ = self.loss(pred, label)
                pos_pred = pred.gather(-1, target.unsqueeze(-1))
                # filter rank
                ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
                return loss, ranking.to(device)
        
    
    def predict(self, batch, all_loss=None, metric=None):
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        device = pos_h_index.device
        batch_size = len(batch.h_id)
        graph = self.get_graph(batch).to(device)
        
        # graph feature
        all_index = torch.arange(graph.num_node, device=device)
        all_kgl_index = self.id2tokenid(all_index, split=batch.split)
        
        if self.training:
            # train
            neg_index = self._strict_negative(
                pos_h_index, pos_t_index, pos_r_index)

            h_index = pos_h_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            t_index = pos_t_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            r_index = pos_r_index.unsqueeze(-1).repeat(2,
                                                       self.num_negative + 1)
            t_index[:batch_size, 1:] = neg_index[:batch_size]
            h_index[batch_size:, 1:] = neg_index[batch_size:]
            
            h_id, r_id, t_id = h_index, r_index, t_index
        else:
            # test all
            h_index, t_index = torch.meshgrid(pos_h_index, all_index)  # batch size x num ent
            # inverse
            it_index, ih_index = torch.meshgrid(pos_t_index, all_index)
            
            r_index = pos_r_index.unsqueeze(-1).expand(-1, len(all_index))
            
            # triplet feature
            h_id = torch.cat([h_index, ih_index])
            r_id = torch.cat([r_index, r_index])
            t_id = torch.cat([t_index, it_index])
            
        # llm feature
        h_kgl_tokenid = torch.cat([batch.h_tokenid, batch.t_tokenid])
        r_kgl_tokenid = torch.cat([batch.r_tokenid, batch.inv_r_tokenid])
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        input_length = batch.input_length
        
        pred = self.llmodel(h_id,
                            r_id,
                            t_id,
                            h_kgl_tokenid,
                            r_kgl_tokenid,
                            graph,
                            all_index,
                            all_kgl_index,
                            input_ids,
                            attention_mask,
                            input_length,
                            )
        return pred
    
    def target(self, batch):
        # test target
        pos_h_index, pos_t_index, pos_r_index = batch.h_id, batch.t_id, batch.r_id
        batch_size = len(batch.h_id)
        graph = self.get_eval_graph(batch)

        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        edge_index, num_t_truth = graph.match(pattern)
        t_truth_index = graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(batch_size, graph.num_node,
                            dtype=torch.bool, device=pos_h_index.device)
        t_mask[pos_index, t_truth_index] = 0

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        edge_index, num_h_truth = graph.match(pattern)
        h_truth_index = graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(batch_size, graph.num_node,
                            dtype=torch.bool, device=pos_h_index.device)
        h_mask[pos_index, h_truth_index] = 0

        mask = torch.cat([t_mask, h_mask])
        target = torch.cat([pos_t_index, pos_h_index])

        return mask, target
        
    def predict_and_target(self, batch, all_loss=None, metric=None):
        return self.predict(batch, all_loss, metric), self.target(batch)

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation
        fact_mask = torch.ones(len(dataset), dtype=torch.bool)
        fact_mask[valid_set.indices] = 0
        fact_mask[test_set.indices] = 0
        self.graph = dataset.graph
        self.fact_graph = dataset.graph.edge_mask(fact_mask)
        return train_set, valid_set, test_set

    def id2tokenid(self, id, split='test', entity=True):
        if entity:
            id2rawname = np.array(self.dataset.kgdata.entity_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n]
                           for n in rawname])
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def get_graph(self, batch):
        return self.fact_graph
    
    def get_eval_graph(self, batch):
        return self.graph

    @torch.no_grad()
    def _strict_negative(self, pos_h_index, pos_t_index, pos_r_index):
        batch_size = len(pos_h_index)
        any = -torch.ones_like(pos_h_index)

        pattern = torch.stack([pos_h_index, any, pos_r_index], dim=-1)
        # pattern = pattern[:batch_size // 2]
        edge_index, num_t_truth = self.fact_graph.match(pattern)
        t_truth_index = self.fact_graph.edge_list[edge_index, 1]
        pos_index = torch.repeat_interleave(num_t_truth)
        t_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        t_mask[pos_index, t_truth_index] = 0
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        neg_t_index = functional.variadic_sample(neg_t_candidate, num_t_candidate, self.num_negative)

        pattern = torch.stack([any, pos_t_index, pos_r_index], dim=-1)
        # pattern = pattern[batch_size // 2:]
        edge_index, num_h_truth = self.fact_graph.match(pattern)
        h_truth_index = self.fact_graph.edge_list[edge_index, 0]
        pos_index = torch.repeat_interleave(num_h_truth)
        h_mask = torch.ones(len(pattern), self.num_entity, dtype=torch.bool, device=self.device)
        h_mask[pos_index, h_truth_index] = 0
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        neg_h_index = functional.variadic_sample(neg_h_candidate, num_h_candidate, self.num_negative)

        neg_index = torch.cat([neg_t_index, neg_h_index])

        return neg_index    


class KGL4IndKGC(KGL4KGC):

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set
        self.num_entity = dataset.num_entity
        self.num_relation = dataset.num_relation

        self.graph = dataset.graph
        self.fact_graph = dataset.fact_graph
        self.inductive_graph = dataset.inductive_graph
        self.inductive_fact_graph = dataset.inductive_fact_graph

    def id2tokenid(self, id, split='test', entity=True):
        if entity:
            if split == 'test':
                id2rawname = np.array(self.dataset.kgdata.inductive_vocab)
            else:
                id2rawname = np.array(self.dataset.kgdata.transductive_vocab)
        else:
            id2rawname = np.array(self.dataset.kgdata.relation_vocab)
            
        rawname = id2rawname[id.cpu()]
        tokenid = np.stack([self.dataset.rawname2tokenid.loc[n]
                           for n in rawname])
        return torch.tensor(tokenid, dtype=id.dtype, device=id.device)

    def get_graph(self, batch):
        return self.inductive_fact_graph if batch.split == "test" else self.fact_graph

    def get_eval_graph(self, batch):
        return self.inductive_graph if batch.split == "test" else self.graph

