import os
import json
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from triplet_mask import construct_mask, construct_self_negative_mask
from dict_hub import get_entity_dict, get_link_graph, get_tokenizer, get_train_triplet_dict, get_rule_dict # get_neighbor_trpilets
from logger_config import logger
import random

entity_dict = get_entity_dict()
train_triplet_dict = get_train_triplet_dict()
rule_dict = get_rule_dict()
if args.use_link_graph:
    # make the lazy data loading happen
    get_link_graph()


def _custom_tokenize(text: str, return_token_type_ids) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_tensors="pt",
                               truncation=True)
    if return_token_type_ids:
        token_type_ids = []
        current_token_type = 0
        for token in encoded_inputs.input_ids[0]:
            token_type_ids.append(current_token_type)
            # 遇到[SEP]时，切换句子的token_type_id
            if token == tokenizer.convert_tokens_to_ids('[SEP]'):
                current_token_type = 1 if current_token_type == 0 else 0

        # 确保token_type_ids的长度与input_ids一致
        assert len(token_type_ids) == len(encoded_inputs.input_ids[0])
        input_ids = encoded_inputs['input_ids'][0]
        mask_h_id = tokenizer.convert_tokens_to_ids('[MASK_h]')
        struct_id = tokenizer.convert_tokens_to_ids('[STRUCT]')
        struct_inices = (input_ids == struct_id).nonzero().squeeze()
        mask_indices_t = (input_ids == tokenizer.mask_token_id).nonzero()
        mask_indices_h = (input_ids == mask_h_id).nonzero()
        mask_indices = torch.cat((mask_indices_t, mask_indices_h), dim=0).squeeze()
        return encoded_inputs, token_type_ids, mask_indices, struct_inices
    else:
        input_ids = encoded_inputs['input_ids']
        mask_indices = (input_ids == tokenizer.mask_token_id).nonzero()
        return encoded_inputs, mask_indices


def _parse_entity_name(entity: str) -> str:
    if args.task.lower() == 'wn18rr':
        # family_alcidae_NN_1
        entity = ' '.join(entity.split('_')[:-2])
        return entity
    # a very small fraction of entities in wiki5m do not have name
    return entity or ''


def _concat_name_desc_struc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        text = '[STRUC] {}: {}'.format(entity, entity_desc)
    else:
        text = '[STRUC] ' + entity
    if '[STRUC] ' not in text:
        print('no struc')
    return text

def _concat_name_desc(entity: str, entity_desc: str) -> str:
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_neighbor_desc(head_id: str, tail_id: str = None) -> str:
    neighbor_ids = get_link_graph().get_neighbor_ids(head_id)
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids = [n_id for n_id in neighbor_ids if n_id != tail_id]
    entities = [entity_dict.get_entity_by_id(n_id).entity for n_id in neighbor_ids]
    entities = [_parse_entity_name(entity) for entity in entities]
    return ' '.join(entities)

def get_neighbor_trpilets(entity, relation, related_entity, num_forward, num_backward):
    if entity in train_triplet_dict.t2hr:
        candidate_triplets = list(set(map(tuple, train_triplet_dict.t2hr[entity])) - {(related_entity, relation)})
        if len(candidate_triplets) >=num_forward:
            forward_triplets = list(random.sample(candidate_triplets, num_forward))   # entity is the tail in the triplet
        elif len(candidate_triplets)>0:
            forward_triplets = list(random.choices(candidate_triplets, k=num_forward))
        else:
            forward_triplets = []
    else:
        forward_triplets = []
    if entity in train_triplet_dict.h2rt:
        candidate_triplets = list(set(map(tuple, train_triplet_dict.h2rt[entity])) - {(relation, related_entity)})
        if len(candidate_triplets) >=num_backward:
            backward_triplets = list(random.sample(candidate_triplets, num_backward))
        elif len(candidate_triplets) > 0:
            backward_triplets = list(random.choices(candidate_triplets, k=num_backward))
        else:
            backward_triplets = []
    else:
        backward_triplets = []

    return forward_triplets, backward_triplets

def get_triplet_seq(triplet_list: str, forward: bool):
    tokenizer = get_tokenizer()
    seqs = []
    if forward:
        for triplet in triplet_list:
            entity = _parse_entity_name(entity_dict.get_entity_by_id(triplet[0]).entity)
            entity_desc = entity_dict.get_entity_by_id(triplet[0]).entity_desc
            entity_text = _concat_name_desc(entity, entity_desc)
            encoded_text = tokenizer(entity_text, add_special_tokens=False)
            encoded_er = tokenizer(text=entity_text + triplet[1], add_special_tokens=False)
            if len(encoded_er['input_ids']) > args.max_num_tokens - 7:
                ent_tailored = encoded_text['input_ids'][:-(len(encoded_er['input_ids']) - args.max_num_tokens + 7)]
                entity_text = tokenizer.decode(ent_tailored, skip_special_tokens=True)
            seq = entity_text + ' [SEP] ' + triplet[1] + ' [SEP] ' + '[MASK]'
            seqs.append(seq)

    else:
        for triplet in triplet_list:
            entity = _parse_entity_name(entity_dict.get_entity_by_id(triplet[1]).entity)
            entity_desc = entity_dict.get_entity_by_id(triplet[1]).entity_desc
            entity_text = _concat_name_desc(entity, entity_desc)
            seq = '[MASK_h]' + ' [SEP] ' + triplet[0] + ' [SEP] ' + entity_text
            seqs.append(seq)

    return seqs

def _custom_tokenize_tri(text: str):
    tokenizer = get_tokenizer()
    if len(text) == 0:
        text = ['[UNK] [UNK] [UNK]' for _ in range(args.num_triplets)]
    if len(text) < args.num_triplets:
        num = len(text)
        text = text + ['[UNK] [UNK] [UNK]' for _ in range(args.num_triplets-num)]
    encoded_inputs = tokenizer(text=text,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_tensors="pt",
                               truncation=True,
                               padding=True)
    token_type_ids = []
    current_token_type = 0
    # 遍历每个句子的 input_ids，手动生成 token_type_ids
    for i, input_id in enumerate(encoded_inputs['input_ids']):
        token_type_id = []
        for token in input_id:
            token_type_id.append(current_token_type)

            # 遇到 [SEP] 时，切换 token_type_id
            if token == tokenizer.convert_tokens_to_ids('[SEP]'):
                current_token_type = 1 if current_token_type == 0 else 0
        token_type_ids.append(token_type_id)
        # 切换到下一个句子的 token_type_id
        current_token_type = 1 if current_token_type == 0 else 0

    input_ids = encoded_inputs['input_ids']
    mask_h_id = tokenizer.encode('[MASK_h]', add_special_tokens=False)[0]
    mask_indices_t = (input_ids == tokenizer.mask_token_id).nonzero()
    mask_indices_h = (input_ids == mask_h_id).nonzero()
    mask_indices = torch.cat((mask_indices_t, mask_indices_h), dim=0)
    if len(mask_indices) == 0:
        mask_indices = torch.zeros(args.num_triplets, 2)
    if len(mask_indices) < args.num_triplets:
        n = len(mask_indices)
        mask_indices = torch.cat([mask_indices,  torch.zeros(args.num_triplets-n, 2)], dim=0)
    return encoded_inputs, token_type_ids, mask_indices


class Example:

    def __init__(self, head_id, relation, tail_id, forward, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.forward = forward

    @property
    def head_desc(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        if args.use_link_graph:
            if len(head_desc.split()) < 20:
                head_desc += ' ' + get_neighbor_desc(head_id=self.head_id, tail_id=self.tail_id)
            if len(tail_desc.split()) < 20:
                tail_desc += ' ' + get_neighbor_desc(head_id=self.tail_id, tail_id=self.head_id)

        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)

        head_encoded_inputs = _custom_tokenize(text=head_text, return_token_type_ids=False)
        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        tail_encoded_inputs = _custom_tokenize(text=tail_text, return_token_type_ids=False)
        tokenizer = get_tokenizer()
        if self.forward:
            head_encoded = tokenizer.tokenize(head_text)
            hr_encoded = tokenizer.tokenize(head_text + self.relation)
            if len(hr_encoded) > args.max_num_tokens - 8:
                ent_tailored = head_encoded[:-(len(hr_encoded) - args.max_num_tokens + 8)]
                ent_tailored_ids = tokenizer.convert_tokens_to_ids(ent_tailored)
                head_text = tokenizer.decode(ent_tailored_ids, skip_special_tokens=False)
            qr_text = '[STRUCT] ' + head_text + ' [SEP] ' + self.relation + ' [SEP]' + ' [MASK]'
            qe_encoded_inputs = head_encoded_inputs
            ae_encoded_inputs = tail_encoded_inputs
            pathes = get_path_with_rule(self.head_id, self.relation,self.tail_id)
            head_forward_triplets, head_backward_triplets = get_neighbor_trpilets(self.head_id, self.relation,
                                                                                  self.tail_id,
                                                                                  args.num_triplets // 2,
                                                                                  args.num_triplets // 2)
            head_triplet_seqs = get_triplet_seq(head_forward_triplets, forward=True) + get_triplet_seq(
                head_backward_triplets, forward=False)
            h_tri_inputs, h_tri_token_type_ids, h_tri_mask_index = _custom_tokenize_tri(text=head_triplet_seqs)
            qe_tri_inputs, qe_tri_token_type_ids, qe_tri_mask_index = h_tri_inputs, h_tri_token_type_ids, h_tri_mask_index

        else:
            tail_encoded = tokenizer.tokenize(tail_text)
            rt_encoded = tokenizer.tokenize(tail_text + self.relation)
            if len(rt_encoded) > args.max_num_tokens - 8:
                ent_tailored = tail_encoded[:-(len(rt_encoded) - args.max_num_tokens + 8)]
                ent_tailored_ids = tokenizer.convert_tokens_to_ids(ent_tailored)
                tail_text = tokenizer.decode(ent_tailored_ids, skip_special_tokens=False)
            qr_text = '[MASK_h]' + ' [SEP] ' + self.relation + ' [SEP] [STRUCT] ' +  tail_text

            qe_encoded_inputs = tail_encoded_inputs
            ae_encoded_inputs = head_encoded_inputs
            tail_forward_triplets, tail_backward_triplets = get_neighbor_trpilets(self.tail_id, self.relation,
                                                                                  self.head_id,
                                                                                  args.num_triplets // 2,
                                                                                  args.num_triplets // 2)

            tail_triplet_seqs = get_triplet_seq(tail_forward_triplets, forward=True) + get_triplet_seq(
                tail_backward_triplets, forward=False)

            t_tri_inputs, t_tri_token_type_ids, t_tri_mask_index = _custom_tokenize_tri(text=tail_triplet_seqs)
            qe_tri_inputs, qe_tri_token_type_ids, qe_tri_mask_index = t_tri_inputs, t_tri_token_type_ids, t_tri_mask_index

        qr_encoded_inputs, qr_token_type_ids, qr_mask_index, qr_struct_index= _custom_tokenize(text=qr_text, return_token_type_ids=True)
        if qr_mask_index.numel() == 0:
            print(qr_text)
        # if not torch.isin(qr_encoded_inputs['input_ids'], torch.tensor([28996])).any():
        #     print('no struc ')

        return {'qr_token_ids': qr_encoded_inputs['input_ids'],
                'qr_token_type_ids': qr_token_type_ids,
                'qr_mask_index': qr_mask_index,
                'qr_struct_index': qr_struct_index,
                'ae_token_ids': ae_encoded_inputs[0]['input_ids'],
                'ae_token_type_ids': ae_encoded_inputs[0]['token_type_ids'],
                'qe_token_ids': qe_encoded_inputs[0]['input_ids'],
                'qe_token_type_ids': qe_encoded_inputs[0]['token_type_ids'],
                'qe_tri_inputs': qe_tri_inputs['input_ids'],
                'qe_tri_token_type_ids': qe_tri_token_type_ids,
                'qe_tri_mask_index': qe_tri_mask_index,
                'obj': self}


class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize()


def load_data(path: str,
              add_forward_triplet: bool = True,
              add_backward_triplet: bool = True,
              ) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_triplet or add_backward_triplet
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]
        if add_forward_triplet:
            obj['forward'] = True
            examples.append(Example(**obj))
        if add_backward_triplet:
            obj['forward'] = False
            examples.append(Example(**obj))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    qr_token_ids, qr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['qr_token_ids'][0]) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    qr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['qr_token_type_ids']) for ex in batch_data],
        need_mask=False)
    for ex in batch_data:
        if ex['qr_mask_index'].numel() != 1:
            print(ex['qr_mask_index'])
    qr_mask_index = torch.LongTensor(
        [int(ex['qr_mask_index']) for ex in batch_data])
    qr_struct_index = torch.LongTensor(
        [int(ex['qr_struct_index']) for ex in batch_data])

    ae_token_ids, ae_mask = to_indices_and_mask(
        [torch.LongTensor(ex['ae_token_ids'][0]) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    ae_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['ae_token_type_ids'][0]) for ex in batch_data],
        need_mask=False)

    qe_token_ids, qe_mask = to_indices_and_mask(
        [torch.LongTensor(ex['qe_token_ids'][0]) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    qe_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['qe_token_type_ids'][0]) for ex in batch_data],
        need_mask=False)

    qe_tri_ids, qe_tri_mask = to_indices_and_mask_tri(
        [torch.LongTensor(ex['qe_tri_inputs']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    qe_tri_token_type_ids = to_indices_and_mask_tri(
        [torch.LongTensor(ex['qe_tri_token_type_ids']) for ex in batch_data],
        need_mask=False)
    qe_tri_mask_index = torch.cat(
        [ex['qe_tri_mask_index'][:,1] for ex in batch_data])

    qe_tri_ids = qe_tri_ids.view(ae_token_ids.size(0), -1, qe_tri_ids.size(1))
    qe_tri_mask = qe_tri_mask.view(ae_token_ids.size(0), -1, qe_tri_mask.size(1))
    qe_tri_token_type_ids = qe_tri_token_type_ids.view(ae_token_ids.size(0), -1, qe_tri_token_type_ids.size(1))
    qe_tri_mask_index = qe_tri_mask_index.view(ae_token_ids.size(0), -1)



    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'qr_token_ids': qr_token_ids,
        'qr_mask': qr_mask,
        'qr_token_type_ids': qr_token_type_ids,
        'qr_mask_index': qr_mask_index,
        'qr_struct_index': qr_struct_index,
        'ae_token_ids': ae_token_ids,
        'ae_mask': ae_mask,
        'ae_token_type_ids': ae_token_type_ids,
        'qe_token_ids': qe_token_ids,
        'qe_mask': qe_mask,
        'qe_token_type_ids': qe_token_type_ids,
        'qe_tri_ids': qe_tri_ids,
        'qe_tri_mask': qe_tri_mask,
        'qe_tri_token_type_ids': qe_tri_token_type_ids,
        'qe_tri_mask_index': qe_tri_mask_index,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices


def to_indices_and_mask_tri(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(1) for t in batch_tensor])
    batch_size = len(batch_tensor)
    if len(batch_tensor[0]) > 1:
        batch_size = batch_size * batch_tensor[0].size(0)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i * t.size(0):(i + 1) * t.size(0), :t.size(1)].copy_(t.squeeze())
        if need_mask:
            mask[i * t.size(0):(i + 1) * t.size(0), :t.size(1)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices