from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig

from triplet_mask import construct_mask
from dict_hub import get_tokenizer
import torch.nn.functional as F


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    loss: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    qr_vector: torch.tensor
    ae_vector: torch.tensor


def aggregate_neighbors_with_attention(query, neighbors):
    """
    计算邻居信息并根据邻居与问题tensor的相似性加权聚合。

    :param query: (batch_size, hidden_dim) 问题tensor
    :param neighbors: (batch_size, num_neighbors, hidden_dim) 邻居tensor
    :return: (batch_size, hidden_dim) 聚合后的邻居信息
    """
    # 计算问题和邻居之间的相似性，使用点积（可以选择其他相似性方法）
    similarity = torch.bmm(neighbors, query.unsqueeze(2))  # (batch_size, num_neighbors, 1)
    similarity = similarity.squeeze(2)  # (batch_size, num_neighbors)

    # 使用 LeakyReLU 激活
    attention_scores = F.leaky_relu(similarity, negative_slope=0.2)  # (batch_size, num_neighbors)

    # 使用 softmax 对注意力系数进行归一化
    attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_neighbors)

    # 根据注意力系数加权邻居信息
    weighted_neighbors = torch.bmm(attention_weights.unsqueeze(1), neighbors)  # (batch_size, 1, hidden_dim)

    # 聚合结果（通过移除多余的维度）
    aggregated_neighbors = weighted_neighbors.squeeze(1)  # (batch_size, hidden_dim)

    return aggregated_neighbors



class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]
        self.tokenizer = get_tokenizer()
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.hr_bert.resize_token_embeddings (len(self.tokenizer))
        self.tail_bert = deepcopy(self.hr_bert)
        self.struct_MLP = nn.Sequential(nn.Linear(768, 768), nn.ReLU())

    def _encode(self, encoder, token_ids, mask, token_type_ids, mask_index=None, struct_vector=None, struct_index=None ):
        if struct_vector is not None:
            embedding_layer = encoder.get_input_embeddings()
            ent_vector = embedding_layer(token_ids)
            # struct_index = torch.ones(ent_vector.size(0), dtype=torch.long)
            ent_vector[torch.arange(ent_vector.size(0)), struct_index, :] = struct_vector.to(torch.float32)
            outputs = encoder(inputs_embeds=ent_vector,
                              attention_mask=mask,
                              token_type_ids=token_type_ids,
                              return_dict=True)
        else:
            outputs = encoder(input_ids=token_ids,
                              attention_mask=mask,
                              token_type_ids=token_type_ids,
                              return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        if mask_index is not None:
            mask_index = mask_index.to(torch.long)
            mask_output = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_index, :]
            mask_output = nn.functional.normalize(mask_output, dim=1)
            return mask_output
        return cls_output

    def hr_encode(self, encoder, token_ids, mask, token_type_ids, mask_index):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        mask_index = mask_index.to(torch.long)
        mask_output = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_index, :]
        mask_output = nn.functional.normalize(mask_output, dim=1)
        return mask_output

    def forward(self, qr_token_ids, qr_mask, qr_token_type_ids, qr_mask_index, qr_struct_index,
                ae_token_ids, ae_mask, ae_token_type_ids,
                qe_token_ids, qe_mask, qe_token_type_ids,
                qe_tri_ids, qe_tri_token_type_ids, qe_tri_mask, qe_tri_mask_index,
                only_ent_embedding=False, **kwargs) -> dict:
        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=ae_token_ids,
                                              tail_mask=ae_mask,
                                              tail_token_type_ids=ae_token_type_ids)

        q_vector = self._encode(self.hr_bert,
                                   token_ids=qr_token_ids,
                                   mask=qr_mask,
                                   token_type_ids=qr_token_type_ids,
                                   mask_index=qr_mask_index,)
        qe_tri_ids = qe_tri_ids.view(-1, qe_tri_ids.size(-1))
        qe_tri_mask = qe_tri_mask.view(-1, qe_tri_mask.size(-1))
        qe_tri_token_type_ids = qe_tri_token_type_ids.view(-1, qe_tri_token_type_ids.size(-1))
        qe_tri_mask_index = qe_tri_mask_index.view(-1)
        qe_tri_vector = self._encode(self.tail_bert,
                                     token_ids=qe_tri_ids,
                                     mask=qe_tri_mask,
                                     token_type_ids=qe_tri_token_type_ids,
                                     mask_index=qe_tri_mask_index)
        qe_tri_vector = qe_tri_vector.view(qe_token_ids.size(0), -1, qe_tri_vector.size(1))
        qe_tri_mask_index = qe_tri_mask_index.view(qe_token_ids.size(0), -1, 1)
        qe_tri_mask_index = (qe_tri_mask_index != 0).int()
        qe_masked = qe_tri_vector * qe_tri_mask_index
        qe_tri_masked_mean = aggregate_neighbors_with_attention(q_vector, qe_masked)
        # qe_tri_masked_mean = self.struct_MLP(qe_tri_masked_mean)
        # qe_tri_masked_mean = nn.functional.normalize(qe_tri_masked_mean, dim=1)
        qr_vector = self._encode(self.hr_bert,
                                 token_ids=qr_token_ids,
                                 mask=qr_mask,
                                 token_type_ids=qr_token_type_ids,
                                 mask_index=qr_mask_index,
                                 struct_vector=qe_tri_masked_mean,
                                 struct_index=qr_struct_index)

        qe_vector = self._encode(self.tail_bert,
                                 token_ids=qe_token_ids,
                                 mask=qe_mask,
                                 token_type_ids=qe_token_type_ids)

        # ae_tri_vector = self._encode(self.hr_bert,
        #                            token_ids=ae_tri_ids,
        #                            mask=ae_tri_mask,
        #                            token_type_ids=ae_tri_token_type_ids,
        #                            mask_index=ae_tri_mask_index)
        # ae_tri_vector = ae_tri_vector.view(qr_token_ids.size(0), -1, ae_tri_vector.size(1))
        # ae_tri_mask_index = ae_tri_mask_index.view(qr_token_ids.size(0), -1, 1)
        # ae_tri_mask_index = (ae_tri_mask_index != 0).int()
        # ae_masked_sum = (ae_tri_vector * ae_tri_mask_index).sum(dim=1)
        # ae_valid_count = ae_tri_mask_index.sum(dim=1)  # valid paths number
        # ae_valid_count = torch.where(ae_valid_count == 0, torch.ones_like(ae_valid_count), ae_valid_count)
        # ae_tri_masked_mean = ae_masked_sum / ae_valid_count

        ae_vector = self._encode(self.tail_bert,
                                 token_ids=ae_token_ids,
                                 mask=ae_mask,
                                 token_type_ids=ae_token_type_ids)




        # DataParallel only support tensor/dict
        return {'qr_vector': qr_vector,
                'ae_vector': ae_vector,
                'qe_vector': qe_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        qr_vector, ae_vector = output_dict['qr_vector'], output_dict['ae_vector']
        batch_size = qr_vector.size(0)
        # labels = torch.zeros(batch_size, dtype=torch.long).to(qr_vector.device)
        labels = torch.arange(batch_size).to(qr_vector.device)

        logits = qr_vector.mm(ae_vector.t())
        all_logits = logits
        triplet_mask = batch_dict.get('triplet_mask', None)
        triplet_mask.fill_diagonal_(False)

        if self.training:
            all_logits = torch.where(~triplet_mask, logits - self.add_margin, logits)
            # logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        pos_logits = torch.where(~triplet_mask, all_logits, torch.tensor(0.0).to(logits.device))
        all_logits *= self.log_inv_t.exp()
        pos_logits *= self.log_inv_t.exp()
        # logits *= self.log_inv_t.exp()
        triplet_mask_all = triplet_mask

        # triplet_mask = batch_dict.get('triplet_mask', None)
        # if triplet_mask is not None:
        #     logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(qr_vector, ae_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        # if self.args.use_self_negative and self.training:
        #     qe_vector = output_dict['qe_vector']
        #     self_neg_logits = torch.sum(qr_vector * qe_vector, dim=1) * self.log_inv_t.exp()
        #     self_negative_mask = batch_dict['self_negative_mask']
        #     self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
        #     logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        if self.args.use_self_negative and self.training:
            qe_vector = output_dict['qe_vector']
            self_neg_logits = torch.sum(qr_vector * qe_vector, dim=1)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits = torch.where(self_negative_mask, self_neg_logits,
                                          self_neg_logits - self.add_margin)* self.log_inv_t.exp()
            all_logits = torch.cat([all_logits, self_neg_logits.unsqueeze(1)], dim=-1)

            # self_neg_logits.masked_fill_(~self_negative_mask, 0)

            self_pos_logits = torch.where(~self_negative_mask, self_neg_logits, torch.tensor(0.0).to(self_neg_logits.device))
            pos_logits = torch.cat([pos_logits, self_pos_logits.unsqueeze(1)], dim=-1)
            # neg_logits = torch.cat([neg_logits, self_neg_logits.unsqueeze(1)], dim=-1)
            triplet_mask_all = torch.cat([triplet_mask, self_negative_mask.unsqueeze(1)], dim=1)



        num_positives_hr = (pos_logits != 0).sum(dim=1)
        num_positives_t = (pos_logits != 0).sum(dim=0)
        pos_logits_hr = torch.log(torch.exp(pos_logits) / torch.sum(torch.exp(all_logits), dim=-1).unsqueeze(1))
        pos_logits_t = torch.log(torch.exp(pos_logits[:, :pos_logits.size(0)]) / torch.sum(torch.exp(all_logits[:, :pos_logits.size(0)]), dim=0).unsqueeze(0))


        pos_logits_hr.masked_fill_(triplet_mask_all, 0)
        pos_logits_t.masked_fill_(triplet_mask, 0)
        loss1 = -(pos_logits_hr.sum(dim=1) / num_positives_hr).mean()
        loss2 = -(pos_logits_t.sum(dim=0) / num_positives_t[:pos_logits.size(0)]).mean()
        loss = loss1 + loss2  # + loss_flat1 -1#+ loss_flat2 -2
        # neg_logits = []
        # pos_logits = []
        # # 遍历每一行，去掉对角线元素
        # for i in range(logits.size(0)):
        #     row = torch.cat([logits[i, :i], logits[i, i + 1:]])  # 去掉第i列
        #     pos_logit = logits[i,i]
        #     neg_logits.append(row)
        #     pos_logits.append(pos_logit)
        #
        # # 将结果列表转换成一个 tensor
        # neg_logits = torch.stack(neg_logits)
        # pos_logits = torch.stack(pos_logits)
        # # negative_adversarial_score = F.softmax(neg_logits, dim=1) * neg_logits.size(1)
        # # x = negative_adversarial_score[0].sum()
        # # neg_logits *= negative_adversarial_score
        # logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)


        return {'logits': logits,
                'loss': loss,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'qr_vector': qr_vector.detach(),
                'ae_vector': ae_vector.detach()}

    # def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
    #     hr_vector, tail_vector = output_dict['qr_vector'], output_dict['ae_vector']
    #     batch_size = hr_vector.size(0)
    #     labels = torch.arange(batch_size).to(hr_vector.device)
    #
    #     logits = hr_vector.mm(tail_vector.t())
    #     # triplet_mask = batch_dict.get('triplet_mask', None)
    #     # triplet_mask.fill_diagonal_(False)
    #
    #     if self.training:
    #         # logits = torch.where(~triplet_mask, logits - self.add_margin, logits)
    #         logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
    #     logits *= self.log_inv_t.exp()
    #
    #     # pos_logits = torch.where(~triplet_mask, logits, torch.tensor(0.0).to(logits.device))
    #
    #
    #
    #     triplet_mask = batch_dict.get('triplet_mask', None)
    #     if triplet_mask is not None:
    #         logits.masked_fill_(~triplet_mask, -1e4)
    #
    #     if self.pre_batch > 0 and self.training:
    #         pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
    #         logits = torch.cat([logits, pre_batch_logits], dim=-1)
    #
    #     if self.args.use_self_negative and self.training:
    #         head_vector = output_dict['qe_vector']
    #         self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
    #         self_negative_mask = batch_dict['self_negative_mask']
    #         self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
    #         logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
    #
    #     return {'logits': logits,
    #             'labels': labels,
    #             'inv_t': self.log_inv_t.detach().exp(),
    #             'qr_vector': hr_vector.detach(),
    #             'ae_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        # ent_tri_vectors = self._encode(self.tail_bert,
        #                                token_ids=t_tri_ids,
        #                                mask=t_tri_mask,
        #                                token_type_ids=t_tri_token_type_ids,
        #                                mask_index=t_tri_mask_index)
        # tail_tri_vector = ent_tri_vectors.view(tail_token_ids.size(0), -1, ent_tri_vectors.size(1))
        # t_tri_mask_index = t_tri_mask_index.view(tail_token_ids.size(0), -1, 1)
        # t_tri_mask_index = (t_tri_mask_index != 0).int()
        # t_masked_sum = (tail_tri_vector * t_tri_mask_index).sum(dim=1)
        # t_valid_count = t_tri_mask_index.sum(dim=1)  # valid paths number
        # t_valid_count = torch.where(t_valid_count == 0, torch.ones_like(t_valid_count), t_valid_count)
        # tail_tri_masked_mean = t_masked_sum / t_valid_count
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
