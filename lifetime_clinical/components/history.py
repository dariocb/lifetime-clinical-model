"""One causal Transformer implementation for all model history summaries."""
import numpy as np
import torch
from torch import nn


class AbsoluteAgeEncoding(nn.Module):
    """Smooth learned encoding of continuous age, without imposed periodicity."""
    def __init__(self,dimension):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,dimension),nn.Tanh(),nn.Linear(dimension,dimension)).double()

    def forward(self,ages_years):
        return self.net(ages_years.reshape(-1,1)/100.)


class CategoricalInputs(nn.Module):
    """Decode schema indicators to IDs, then use small trainable embeddings.

    All-zero groups mean not applicable and use a fixed zero padding embedding.
    Missing/unseen categories retain their declared IDs and explicit flags.
    Existing data contracts can retain indicators; neural layers receive vectors.
    """
    def __init__(self,input_dim,groups=(),drop_columns=()):
        super().__init__()
        self.groups=tuple(groups)
        excluded=set(drop_columns)
        for start,end,_ in groups: excluded.update(range(start,end))
        self.numeric=tuple(i for i in range(input_dim) if i not in excluded)
        self.embeddings=nn.ModuleList([nn.Embedding(end-start+1,size,padding_idx=0).double()
                                      for start,end,size in groups])
        self.output_dim=len(self.numeric)+sum(size for _,_,size in groups)

    def forward(self,x):
        parts=[x[...,list(self.numeric)]]
        for (start,end,_),embedding in zip(self.groups,self.embeddings):
            indicators=x[...,start:end]
            ids=torch.where(indicators.sum(-1)>0,indicators.argmax(-1)+1,
                            torch.zeros(indicators.shape[:-1],dtype=torch.long,device=x.device))
            parts.append(embedding(ids))
        return torch.cat(parts,dim=-1)


def clinical_feature_names(kind,dimension):
    """Recover canonical schemas, including optional EHR vector dimensions."""
    from ..data.preprocessing import ClinicalPreprocessor
    base={"static":20,"history":31,"dynamic":55}[kind]
    extra=dimension-base
    if extra<0 or (kind=="dynamic" and extra%2): return None
    pre=ClinicalPreprocessor((),{},ehr_dimension=extra//2 if kind=="dynamic" else extra)
    if kind=="dynamic":
        from ..data.tutorial import context_feature_names
        return context_feature_names(pre)
    return list(pre.static_feature_names if kind=="static" else pre.token_feature_names)


class NeuralInputs(nn.Module):
    """Age only through learned encodings; categorical IDs through embeddings."""
    def __init__(self,input_dim,hidden_dim,feature_names=None,position_only=False):
        super().__init__()
        names=list(feature_names) if feature_names is not None else [f"value_{i}" for i in range(input_dim)]
        if len(names)!=input_dim: raise ValueError("Feature schema dimension mismatch")
        self.age_columns=tuple(i for i,n in enumerate(names) if n=="age_scaled" or n.endswith("_age_scaled"))
        groups=[];start=0
        sizes={"sex":2,"marital_status":3,"employment":2,"kind":2,
               "event":2,"last_event":2,"variable":3,"output":2}
        while start<len(names):
            key=names[start].split("=")[0]
            if "=" not in names[start]: start+=1;continue
            end=start+1
            while end<len(names) and names[end].startswith(key+"="):end+=1
            groups.append((start,end,sizes.get(key,2)));start=end
        self.categories=CategoricalInputs(input_dim,groups,self.age_columns)
        self.hidden_dim,self.position_only=hidden_dim,position_only
        self.age_encoders=nn.ModuleList([] if position_only else
                          [AbsoluteAgeEncoding(hidden_dim) for _ in self.age_columns])
        self.output_dim=self.categories.output_dim+(0 if position_only else hidden_dim*len(self.age_columns))

    def forward(self,x):
        result=self.categories(x)
        if not self.position_only:
            parts=[result]
            for column,encoder in zip(self.age_columns,self.age_encoders):
                shape=x.shape[:-1]
                parts.append(encoder(x[...,column].reshape(-1)*100)
                             .reshape(*shape,self.hidden_dim))
            result=torch.cat(parts,dim=-1)
        return result


class CausalHistoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads=2, layers=1,
                 position_column=None,position_scale=100.,categorical_groups=(),feature_names=None):
        super().__init__()
        if input_dim<1 or hidden_dim<1 or heads<1 or layers<1 or hidden_dim%heads:
            raise ValueError("Positive dimensions; hidden_dim must be divisible by heads")
        self.input_dim,self.hidden_dim=int(input_dim),int(hidden_dim)
        self.position_column,self.position_scale=position_column,float(position_scale)
        if feature_names is not None:
            self.input_encoder=NeuralInputs(input_dim,hidden_dim,feature_names,position_only=True)
            if position_column is None and self.input_encoder.age_columns:
                self.position_column=self.input_encoder.age_columns[-1]
        else:
            self.input_encoder=CategoricalInputs(input_dim,categorical_groups,
                             () if position_column is None else (position_column,))
        self.projection=nn.Linear(self.input_encoder.output_dim,hidden_dim).double()
        self.start=nn.Parameter(torch.zeros(hidden_dim,dtype=torch.float64))
        self.position_encoder=AbsoluteAgeEncoding(hidden_dim)
        layer=nn.TransformerEncoderLayer(hidden_dim,heads,4*hidden_dim,dropout=0.,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,layers).double()

    def states(self, tokens, positions=None):
        p=self.start
        x=torch.as_tensor(tokens,dtype=p.dtype,device=p.device)
        if x.ndim!=2 or x.shape[1]!=self.input_dim or not torch.isfinite(x).all():
            raise ValueError("Expected finite [history_length, input_dim] tokens")
        sequence=torch.cat([p[None,:],self.projection(self.input_encoder(x))],dim=0)
        positions=(torch.cat([p.new_zeros(1),torch.as_tensor(positions,dtype=p.dtype,device=p.device)])
                   if positions is not None else
                   torch.arange(len(sequence),dtype=p.dtype,device=p.device)
                   if self.position_column is None else
                   torch.cat([p.new_zeros(1),x[:,self.position_column]*self.position_scale]))
        if positions.shape!=(len(sequence),) or not torch.isfinite(positions).all():
            raise ValueError("One finite absolute position per history token required")
        positional=self.position_encoder(positions)
        mask=torch.triu(torch.ones(len(sequence),len(sequence),dtype=torch.bool,device=p.device),1)
        return self.transformer((sequence+positional)[None,:,:],mask=mask)[0]

    def forward(self, tokens, positions=None):
        return self.states(tokens,positions)[-1]
