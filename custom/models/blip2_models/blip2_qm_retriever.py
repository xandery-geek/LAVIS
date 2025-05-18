"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import time
import datetime
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from transformers.modeling_outputs import ModelOutput

from lavis.common.registry import registry
import lavis.common.dist_utils as dist_utils
from lavis.models.base_model import concat_all_gather
from lavis.models.blip_models.blip_outputs import BlipOutputFeatures
from custom.models.blip2_models.blip2 import Blip2Base, disabled_train


class Blip2QMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_itc: Optional[torch.FloatTensor] = None
    loss_kl: Optional[torch.FloatTensor] = None


@registry.register_model("blip2_qm_retriever")
class Blip2QMRetriever(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "qm_retriever": "configs/models/blip2/blip2_qm_retriever.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        rephrase_query=False,
        loss_kl_weight=0.1,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 
            self.visual_encoder.num_features, 
            cross_attention_freq,
            question_encoder=rephrase_query,
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self.max_txt_len = max_txt_len
        self.rephrase_query = rephrase_query
        self.loss_kl_weight = loss_kl_weight

    def forward_for_feature(self, image, text, use_question_encoder=False):
        image_embeds = self.ln_vision(self.visual_encoder(image)) # [batch_size*2, num_embed, embed_dim]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        ) # [batch_siz*2, num_embed]

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [batch_size*2, num_query_tokens, embed_dim]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device) # [batch_size*2, num_query_tokens]

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device) # [batch_size*2, max_txt_len]

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1) # [batch_size, num_query_tokens + max_txt_len]
        
        output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_question_encoder=use_question_encoder,
        )

        multimodal_feats = output.last_hidden_state[:, : query_tokens.size(1), :]
        return multimodal_feats
    
    def forward(self, samples):
        question = samples["question"]
        query_image = samples["query_image"]

        evidence = samples["evidence"]
        evidence_image = samples["evidence_image"]

        bs = query_image.size(0)
        device = query_image.device

        if self.rephrase_query:
            rephrase_question = samples["rephrase_question"]
            rephrase_image = query_image.clone()

            query_feats = self.forward_for_feature(query_image, question, use_question_encoder=True)
            evidence_feats = self.forward_for_feature(evidence_image, evidence, use_question_encoder=False)
            rephrase_query_feats = self.forward_for_feature(rephrase_image, rephrase_question, use_question_encoder=False)
            
            loss_kl = self.loss_kl_weight * F.kl_div(
                F.log_softmax(F.normalize(query_feats, dim=-1), dim=-1),
                F.softmax(F.normalize(rephrase_query_feats, dim=-1), dim=-1),
                reduction='batchmean'
                )

            query_feats = F.normalize(query_feats.mean(dim=1), dim=-1)
            evidence_feats = F.normalize(evidence_feats.mean(dim=1), dim=-1)
        else:
            loss_kl = torch.tensor(0.0).to(device)

            image = torch.cat([query_image, evidence_image], dim=0)
            text = question + evidence

            multimodal_feats = self.forward_for_feature(image, text, use_question_encoder=False)
            pooling_feats = F.normalize(multimodal_feats.mean(dim=1), dim=-1) # [batch_size*2, embed_dim]

            query_feats = pooling_feats[:bs] # [batch_size, embed_dim]
            evidence_feats = pooling_feats[bs:] # [batch_size, embed_dim]

        ###============== Contrastive Loss ===================###
        query_feats_all = concat_all_gather(query_feats) # [batch_size*num_gpu, embed_dim]
        evidence_feat_all = concat_all_gather(evidence_feats) # [batch_size*num_gpu, embed_dim]

        # query-evidence similarity
        sim_q2e = query_feats @ evidence_feat_all.t() # [batch_size, batch_size*num_gpu]
        sim_q2e = sim_q2e / self.temp

        # evidence-query similarity
        sim_e2q = evidence_feats @ query_feats_all.t() # [batch_size, batch_size*num_gpu]
        sim_e2q = sim_e2q / self.temp

        rank = dist.get_rank()
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(device)
    
        loss_itc = (
            F.cross_entropy(sim_q2e, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_e2q, targets, label_smoothing=0.1)
        ) / 2
            
        return Blip2QMOutput(
            loss=loss_itc + loss_kl,
            loss_itc=loss_itc,
            loss_kl=loss_kl,
        )

    @torch.no_grad()
    def extract_features(self, samples, text_type="question"):
        assert text_type in ["question", "evidence"], f"Invalid text type: {text_type}"
        use_question_encoder = (text_type == "question")

        image = samples.get("image")
        text_input = samples.get("text_input")
        
        # return multimodel query features
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text = self.tokenizer(text_input, return_tensors="pt", padding=True).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

        output = self.Qformer.bert(
            text.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            use_question_encoder=use_question_encoder,
        )

        multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]
        return multimodal_embeds

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        max_txt_len = cfg.get("max_txt_len", 32)
        rephrase_query = cfg.get("rephrase_query", False)
        loss_kl_weight = cfg.get("loss_kl_weight", 0.1)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            rephrase_query=rephrase_query,
            loss_kl_weight=loss_kl_weight,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_accuracy(self, data_loader, task_cfg):
        """
        Compute accuracy for the model on the given data loader.
        """

        return compute_accuracy(model=self, data_loader=data_loader)


def compute_accuracy(model, data_loader, **kwargs):
    logging.info("Computing features for evaluation...")
    start_time = time.time()

    query_feats, evidence_feats = [], []
    for samples in data_loader:
        question = samples["question"]
        query_image = samples["query_image"]
        evidence = samples["evidence"]
        evidence_image = samples["evidence_image"]

        query_sample = {"image": query_image, "text_input": question}
        query_feat = model.extract_features(query_sample, text_type="question")
        query_feat = F.normalize(query_feat.mean(dim=1), dim=-1)
        query_feats.append(query_feat)

        evidence_sample = {"image": evidence_image, "text_input": evidence}
        evidence_feat = model.extract_features(evidence_sample, text_type="evidence")
        evidence_feat = F.normalize(evidence_feat.mean(dim=1), dim=-1)
        evidence_feats.append(evidence_feat)
    
    query_feats = torch.cat(query_feats, dim=0)
    evidence_feats = torch.cat(evidence_feats, dim=0)

    sim_q2e = query_feats @ evidence_feats.t()
    max_sim_idx = sim_q2e.argmax(dim=1)
    targets = torch.arange(query_feats.size(0), dtype=torch.long).to(query_feats.device)
    acc = (max_sim_idx == targets).float().mean()

    if dist_utils.is_dist_avail_and_initialized():
        dist.barrier()
        torch.distributed.all_reduce(
            acc, op=torch.distributed.ReduceOp.AVG
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info("Evaluation time {}".format(total_time_str))
    return acc.item()
