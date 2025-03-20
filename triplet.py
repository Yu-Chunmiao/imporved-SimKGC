import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque

from logger_config import logger


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}
        self.rt2heads = {}
        self.h2rt = {}
        self.t2hr = {}
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        for ex in examples:
            self.relations.add(ex['relation'])
            hr_key = (ex['head_id'], ex['relation'])
            rt_key = (ex['relation'], ex['tail_id'])
            if hr_key not in self.hr2tails:
                self.hr2tails[hr_key] = set()
            self.hr2tails[hr_key].add(ex['tail_id'])
            if rt_key not in self.rt2heads:
                self.rt2heads[rt_key] = set()
            self.rt2heads[rt_key].add(ex['head_id'])
            if ex['head_id'] not in self.h2rt:
                self.h2rt[ex['head_id']] = set()
            self.h2rt[ex['head_id']].add((ex['relation'], ex['tail_id']))
            if ex['tail_id'] not in self.t2hr:
                self.t2hr[ex['tail_id']] = set()
            self.t2hr[ex['tail_id']].add((ex['head_id'], ex['relation']))
        self.triplet_cnt = len(examples)

    def get_neighbors(self, e: str, r: str, forward) -> set:
        if forward:
            return self.hr2tails.get((e, r), set())
        else:
            return self.rt2heads.get((r,e), set())



class EntityDict:

    def __init__(self, entity_dict_dir: str, inductive_test_path: str = None):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        if inductive_test_path:
            examples = json.load(open(inductive_test_path, 'r', encoding='utf-8'))
            valid_entity_ids = set()
            for ex in examples:
                valid_entity_ids.add(ex['head_id'])
                valid_entity_ids.add(ex['tail_id'])
            self.entity_exs = [ex for ex in self.entity_exs if ex.entity_id in valid_entity_ids]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)

class Ruledict():
    def __int__(self, rule_dict_path: str):
        self.path = rule_dict_path
        self.r2rules = dict()
        with open(self.path, 'r') as fi:
            for line in fi:
                rule = line.strip().split()
                if len(rule) <= 1:
                    continue
                rule = [int(_) for _ in rule]
                relation = rule[1]
                self.r2rules[relation].append(rule)



class LinkGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build link graph from {}'.format(train_path))
        # id -> set(id)
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = set()
            self.graph[head_id].add(tail_id)
            if tail_id not in self.graph:
                self.graph[tail_id] = set()
            self.graph[tail_id].add(head_id)
        logger.info('Done build link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])


