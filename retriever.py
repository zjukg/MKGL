import torch
from torch import nn
import torch.nn.functional as F
from torchdrug import core
from gnn.model import *




class BasePNARetriever(nn.Module): 
    '''
    Retrieve text information
    '''

    
    def __init__(self, config, text_embeddings, kgl2token, orig_vocab_size):
        super().__init__()
        self.config = config
        self.text_embeddings =text_embeddings
        self.kgl2token = kgl2token
        self.orig_vocab_size = orig_vocab_size
        
        self.down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)
        
        if self.config.text_encoder == 'pna':
            self.re_scaling = nn.Linear(config.r*12, self.config.r)
    
    
    def aggregate_text(self, token_ids, text_embeddings, method='pna'):
        device = text_embeddings.device
        
        token_ids = token_ids.to(device) # Batch x Length
        token_mask = (token_ids > 0).unsqueeze(-1).to(device) # B x L X 1
        token_lengths = token_mask.half().sum(axis=1).to(device) # B X 1
        degree = token_lengths
        token_embs = text_embeddings[token_ids] # B x L x Hidden
        
        mean = (token_embs * token_mask).sum(axis=1) / token_lengths
        if method == 'mean':
            result = mean
        else:
            sq_mean = (token_embs**2 * token_mask).sum(axis=1) / \
                token_lengths
            max, _ = (token_embs*token_mask).max(axis=1)
            min, _ = (token_embs*token_mask).min(axis=1)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat(
                [mean, max, min, std], dim=-1)
            
            scale = degree.log()
            scale = scale / scale.mean()
            scales = torch.cat(
                [torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            
            result = (features.unsqueeze(-1) *
                      scales.unsqueeze(-2)).flatten(-2)

        return result 
    
    def retrieve_text(self, token_ids):
        # token_ids: num_kgl_tokens x num_tokens
        R = self.down_scaling(self.text_embeddings)
        
        result = self.aggregate_text(token_ids, R, self.config.text_encoder)
        
        if self.config.text_encoder == 'pna':
            result = self.re_scaling(result)
        
        return self.norm(result)

    def norm(self, x):
        return F.normalize(x, p=2, dim=1)
                
    
    def forward(self, kgl_ids=None):
        if kgl_ids is not None:
            kgl_ids = kgl_ids - self.orig_vocab_size
            token_ids = self.kgl2token[kgl_ids.cpu()]
        else:
            token_ids = self.kgl2token
        return self.retrieve_text(token_ids)
        


class ContextRetriever(BasePNARetriever):

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.up_scaling = nn.Linear(
                self.config.r, self.config.llm_hidden_dim, bias=False, dtype=torch.float)

    def forward(self, kgl_ids, graph, all_index, all_kgl_index):
        text_embs = super().forward(kgl_ids)
        context = self.up_scaling(text_embs)
        return context
        

        
class ScoreRetriever(BasePNARetriever):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.kg_retriever = core.Configurable.load_config_dict(config.kg_encoder)
        # self.kg_retriever = ConditionedPNA(config.kg_encoder)
        self.h_down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)
        self.r_down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)

    def forward(self, h_id, r_id, t_id,  hidden_states, rel_hidden_states, graph, all_index, all_kgl_index):
        score_text_embs = super().forward(all_kgl_index)
        head_embeds = self.h_down_scaling(hidden_states) 
        rel_embeds = self.r_down_scaling(rel_hidden_states) 
        score = self.kg_retriever(h_id, r_id, t_id, head_embeds, rel_embeds, graph, score_text_embs, all_index)
        
        return score

class RelScoreRetriever(BasePNARetriever):
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.r_down_scaling = nn.Linear(
                self.config.llm_hidden_dim, self.config.r, bias=False, dtype=torch.float)

    def forward(self, rel_hidden_states, all_rel_kgl_index):
        score_text_embs = super().forward(all_rel_kgl_index) # num rel, r
        rel_embeds = self.r_down_scaling(rel_hidden_states) # batch size, r
        score = F.linear(rel_embeds, score_text_embs)
        return score