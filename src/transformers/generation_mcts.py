# Adopted from PPL-MCTS:
# https://github.com/NohTow/PPL-MCTS/blob/a8e9c863c73c43363184c402023f3b623fb9401e/teammates/mcts_ag_bert_uni.py

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm


def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len - length:, ...] = tensor
        else:
            out_tensor[max_len - length:, i, ...] = tensor
    return out_tensor


def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    B = len(sequences)
    out = [[None for kv in range(len(sequences[0][layer]))] for layer in range(len(sequences[0]))]
    for layer in range(len(sequences[0])):
        for kv in range(len(sequences[0][layer])):
            if kv < 2:
                # decoder
                current_shape = sequences[0][layer][kv].shape
                new_shape = (B, current_shape[0], max_len, current_shape[2])
                out[layer][kv] = sequences[0][layer][kv].new_full(new_shape, padding_value)
                for b in range(B):
                    length = sequences[b][layer][kv].shape[1]
                    out[layer][kv][b, :, max_len - length:, ...] = sequences[b][layer][kv]
            else:
                # encoder - no padding needed
                out[layer][kv] = torch.stack([sequences[b][layer][kv] for b in range(B)])
    return out


class NumpyMCTS:
    def __init__(self, root_fun, rec_fun, batch_size, num_simulations, num_actions, num_sparse_actions, pb_c_init,
                 lm_device, lm_caching=True, classifier_caching=False, argpartition_reproducibility=False):
        self._lm_device = lm_device
        self._lm_caching = lm_caching
        self._argpartition_reproducibility = argpartition_reproducibility
        self._batch_size = batch_size
        self._num_simulations = num_simulations
        self._num_actions = num_actions
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        self._pb_c_init = pb_c_init
        self._classifier_caching = classifier_caching
        if classifier_caching:
            raise NotImplementedError(
                "Classifier caching is not implemented. To implement it, note that the MCTS needs to be extended in such a way that it keeps track about both tokenizers and both token_ids. Also, caching for encoder-only or decoder-only models is implemented at the moment, which could be extended to decoder-encoder models with a bit of work and to general classifiers with some more work.")

        self._root_fun = root_fun  # a function called at the root
        self._rec_fun = rec_fun  # a function called in the tree
        # self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        # self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)
        self._labels = torch.zeros((batch_size, 4), dtype=torch.uint8, device=lm_device)

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)  # (B, )
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        self._original_states = {}
        self._classi_states = {}
        self._original_token_ids = {}
        self._original_attention_mask = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._original_states = {}
        self._classi_states = {}
        self._original_token_ids = {}  # Indexed by tuples (batch index, node index)
        self._original_attention_mask = {}

    def set_labels(self, labels):
        self._labels = labels

    def search(
            self,
            original_tokens_ids,
            original_attention_mask,
            pad_token_id,
            eos_token_id,
            max_input_sequence_length,
            tokens_to_generate,
            simulation_callbacks=None,
    ):
        self._reset_tree()

        # Evaluate the root.
        likelihoods = np.array([1.0 for _ in self._batch_range])
        prior, values, states, classi_states = self._root_fun(self, original_tokens_ids, original_attention_mask,
                                                              self._labels, likelihoods)

        # self._adaptive_min_values = 1
        # self._adaptive_max_values = 1 + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, values, states, original_tokens_ids, original_attention_mask,
                         classi_states, np.full(self._batch_size, False, dtype=bool))

        # Do simulations, expansions, and backwards.
        leaf_indices = np.zeros((self._batch_size), np.int32)
        existing_nodes = 0
        tokens_pbar = tqdm(total=tokens_to_generate, desc="Tokens generated")
        for i in range(tokens_to_generate):
            # torch.cuda.empty_cache()
            if simulation_callbacks is not None:
                for simulation_callback in simulation_callbacks:
                    simulation_callback(self, i, -1)

            for sim in range(self._num_simulations):
                node_indices, actions = self.simulate()
                next_node_index = sim + 1 + existing_nodes  # root is 0, therefore we offset by 1.
                self.expand(node_indices, actions, next_node_index, pad_token_id, eos_token_id,
                            max_input_sequence_length)
                leaf_indices.fill(next_node_index)
                self.backward(leaf_indices)

                if simulation_callbacks is not None:
                    for simulation_callback in simulation_callbacks:
                        simulation_callback(self, i, sim)

            visit_counts, _ = self.dense_visit_counts()
            existing_nodes = np.amax(visit_counts)
            # Create new tree with selected node as root
            num_nodes = self._num_simulations + existing_nodes + 1
            batch_node = (self._batch_size, num_nodes)
            temp_visit_counts = np.zeros(batch_node, dtype=np.int32)
            temp_values = np.zeros(batch_node, dtype=np.float32)
            temp_likelihoods = np.zeros(batch_node, dtype=np.float32)
            temp_raw_values = np.zeros(batch_node, dtype=np.float32)
            temp_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_action_from_parents = np.full(batch_node, -1, dtype=np.int32)
            temp_depth = np.zeros(batch_node, dtype=np.int32)
            temp_is_terminal = np.full(batch_node, False, dtype=bool)
            batch_node_action = (self._batch_size, num_nodes, self._num_sparse_actions)  # (B, )
            temp_topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
            temp_children_index = np.full(batch_node_action, -1, dtype=np.int32)
            temp_children_prior = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_values = np.zeros(batch_node_action, dtype=np.float32)
            temp_children_visits = np.zeros(batch_node_action, dtype=np.int32)
            temp_original_states = {}
            temp_classi_states = {}
            temp_original_token_ids = {}  # Indexed by tuples (batch index, node index)
            temp_original_attention_mask = {}

            for b, new_root_action in enumerate(np.argmax(visit_counts, axis=1)):
                new_root_id = self._children_index[b, 0, new_root_action]
                new_node_id = 1
                old_to_new_id = {new_root_id: 0}
                children_to_explore = self._children_index[b, new_root_id][
                    self._children_index[b, new_root_id] != -1].tolist()
                while (len(children_to_explore) > 0):
                    child_id = children_to_explore.pop(0)
                    old_to_new_id[child_id] = new_node_id
                    children_to_explore += self._children_index[b, child_id][
                        self._children_index[b, child_id] != -1].tolist()
                    new_node_id += 1
                for old_id, new_id in old_to_new_id.items():
                    if (new_id != 0):
                        temp_parents[b, new_id] = old_to_new_id[self._parents[b, old_id]]
                        temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    for i, children in enumerate(self._children_index[b, old_id]):
                        if (children != -1):
                            temp_children_index[b, new_id, i] = old_to_new_id[children]
                    temp_visit_counts[b, new_id] = self._visit_counts[b, old_id]
                    temp_values[b, new_id] = self._values[b, old_id]
                    temp_likelihoods[b, new_id] = self._likelihoods[b, old_id]
                    temp_raw_values[b, new_id] = self._raw_values[b, old_id]

                    temp_action_from_parents[b, new_id] = self._action_from_parents[b, old_id]
                    temp_depth[b, new_id] = self._depth[b, old_id] - 1
                    temp_is_terminal[b, new_id] = self._is_terminal[b, old_id]

                    temp_topk_mapping[b, new_id] = self._topk_mapping[b, old_id]
                    temp_children_prior[b, new_id] = self._children_prior[b, old_id]
                    temp_children_values[b, new_id] = self._children_values[b, old_id]
                    temp_children_visits[b, new_id] = self._children_visits[b, old_id]

                    if self._lm_caching:
                        temp_original_states[(b, new_id)] = self._original_states[(b, old_id)]
                    if self._classifier_caching:
                        temp_classi_states[(b, new_id)] = self._classi_states[(b, old_id)]
                    temp_original_token_ids[(b, new_id)] = self._original_token_ids[(b, old_id)]
                    temp_original_attention_mask[(b, new_id)] = self._original_attention_mask[(b, old_id)]

                if self._lm_caching:
                    temp_original_states[(b, 0)] = [
                        [
                            torch.cat((self._original_states[(b, 0)][layer][kv],
                                       self._original_states[(b, new_root_id)][layer][kv]), dim=1)
                            for kv in range(len(self._original_states[(b, 0)][layer]))[:2]
                        ] + [
                            self._original_states[(b, 0)][layer][kv]
                            for kv in range(len(self._original_states[(b, 0)][layer]))[2:]
                        ]
                        for layer in range(len(self._original_states[(b, 0)]))
                    ]
                if self._classifier_caching:
                    temp_classi_states[(b, 0)] = torch.cat(
                        (self._classi_states[(b, 0)], self._classi_states[(b, new_root_id)]), 3)

            self._num_nodes = num_nodes
            self._visit_counts = temp_visit_counts
            self._values = temp_values
            self._likelihoods = temp_likelihoods
            self._raw_values = temp_raw_values
            self._parents = temp_parents
            self._action_from_parents = temp_action_from_parents
            # The 0-indexed depth of the node. The root is the only 0-depth node.
            # The depth of node i, is the depth of its parent + 1.
            self._depth = temp_depth
            self._is_terminal = temp_is_terminal
            self._topk_mapping = temp_topk_mapping
            self._children_index = temp_children_index
            self._children_prior = temp_children_prior
            self._children_values = temp_children_values
            self._children_visits = temp_children_visits
            self._original_states = temp_original_states
            self._original_token_ids = temp_original_token_ids
            self._original_attention_mask = temp_original_attention_mask
            self._classi_states = temp_classi_states
            tokens_pbar.update(1)
            # If every sequences is finished, stop
            if (self._is_terminal[:, 0].all()):
                break

        return torch.stack([self._original_token_ids[(b, 0)] for b in range(self._batch_size)]).to(self._lm_device)

    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        return root_visit_counts, dense_visit_counts

    # def dense_scores(self):
    #     root_index = 0
    #     root_scores = self._children_values[:, root_index, :]
    #     dense_root_scores = np.zeros((self._batch_size, self._num_actions))
    #     dense_root_scores[self._batch_range[:, None], self._child_prob_mapping[:, root_index, :]] = root_scores
    #     root_visit_counts = self._children_visits[:, root_index, :]
    #     return dense_root_scores
    #
    # def dense_mean_scores(self):
    #     root_index = 0
    #     root_visit_counts = self._children_visits[:, root_index, :]
    #     root_scores = self._children_values[:, root_index, :]
    #     root_mean_scores = root_scores / root_visit_counts
    #     dense_mean_scores = np.zeros((self._batch_size, self._num_actions))
    #     dense_mean_scores[self._batch_range[:, None], self._child_prob_mapping[:, root_index, :]] = root_mean_scores
    #     return dense_mean_scores

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            actions = self.uct_select_action(node_indices)
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            is_unexplored = next_node_indices == -1
            if is_unexplored.all():
                return node_indices, actions
            else:
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)

    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :]  # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :]  # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :]  # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices]  # (B)
        node_policy_score = (
                np.sqrt(node_visits[:, None])
                * self._pb_c_init
                * node_children_prior
                / (node_children_visits + 1)
        )
        # (B, A)

        node_value_score = node_children_values

        node_uct_score = node_value_score + node_policy_score  # (B, A)
        actions = np.argmax(node_uct_score, axis=1)
        return actions

    # return state
    def get_original_states_from_node(self, b, n, d):
        original_state_array = [None] * d
        original_state_array[d - 1] = self._original_states[(b, n)]
        while n != 0:
            n = self._parents[(b, n)]
            d -= 1
            original_state_array[d - 1] = self._original_states[(b, n)]
        result = [
            [
                # decoder
                torch.cat([original_state[layer][kv] for original_state in original_state_array], dim=1)
                for kv in range(len(original_state_array[0][layer]))[:2]
            ] + [
                # encoder
                original_state_array[0][layer][kv]
                for kv in range(len(original_state_array[0][layer]))[2:]
            ]
            for layer in range(len(original_state_array[0]))
        ]
        return result

    def get_classi_states_from_node(self, b, n, d):
        # classi_state_array = [None] * d
        # classi_state_array[d - 1] = self._classi_states[(b, n)]
        # while n != 0:
        #     n = self._parents[(b, n)]
        #     d -= 1
        #     classi_state_array[d - 1] = self._classi_states[(b, n)]
        # return torch.cat(classi_state_array, 3)
        raise NotImplementedError()

    def expand(self, node_indices, actions, next_node_index, pad_token_id, eos_token_id, max_input_sequence_length):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""
        # Retrieve token ids and masks for nodes to be evaluated.
        original_tokens_ids = pad_sequences_to_left(
            [self._original_token_ids[(b, n)] for b, n in enumerate(node_indices)], True, pad_token_id)
        original_attention_masks = pad_sequences_to_left(
            [self._original_attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)
        depths = torch.tensor([self._depth[(b, n)] + 1 for b, n in enumerate(node_indices)], device=self._lm_device)
        children_priors = np.array([self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        previous_values = np.array([self._values[(b, n)] for b, n in enumerate(node_indices)])
        previous_node_is_terminal = self._is_terminal[self._batch_range, node_indices[self._batch_range]]  # (B)

        if self._lm_caching:
            original_states_list = pad_sequences_to_left_states(
                [self.get_original_states_from_node(b, n.item(), depths[b].item()) for b, n in enumerate(node_indices)],
                0, max_len=len(original_tokens_ids[0])
            )
        else:
            original_states_list = None
        if self._classifier_caching:
            classi_states_tensor = pad_sequences_to_left_states([
                self.get_classi_states_from_node(b, n, depths[b].item()) for b, n in enumerate(node_indices)], 0,
                max_len=len(original_tokens_ids[0]))
        if (len(original_tokens_ids[0]) >= max_input_sequence_length):
            previous_node_is_terminal[
                torch.sum(original_attention_masks, axis=1).cpu().numpy() >= max_input_sequence_length] = True
            original_tokens_ids = original_tokens_ids[:, -(max_input_sequence_length - 1):]
            original_attention_masks = original_attention_masks[:, -(max_input_sequence_length - 1):]
            if self._lm_caching:
                original_states_list = [
                    [
                        # decoder
                        original_states_list[layer][kv][:, :, -(max_input_sequence_length - 1):]
                        for kv in range(len(original_states_list[layer]))[:2]
                    ] + [
                        # encoder - not affected by max_sequence_limit
                        # TODO verify this is correct
                        original_states_list[layer][kv]
                        for kv in range(len(original_states_list[layer]))[2:]
                    ]
                    for layer in range(len(original_states_list))
                ]
            if self._classifier_caching:
                classi_states_tensor = classi_states_tensor[:, :, :, :, -(max_input_sequence_length - 1):]

        original_states = original_states_list
        if self._classifier_caching:
            classi_states = tuple(tuple(type_of_value for type_of_value in layer) for layer in classi_states_tensor)
        else:
            classi_states = None

        # Convert sparse actions to dense actions for network computation
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        dense_actions[previous_node_is_terminal] = pad_token_id
        # Add actions to list of tokens and extend attention mask by 1
        original_tokens_ids = torch.cat((original_tokens_ids, torch.unsqueeze(
            torch.tensor(dense_actions, dtype=torch.long, device=self._lm_device), 1)), dim=1)
        original_attention_masks = torch.cat((original_attention_masks, torch.unsqueeze(
            torch.ones(len(dense_actions), dtype=torch.long, device=self._lm_device), 1)), dim=1)

        # Check if expanded nodes are terminal
        expanded_node_is_terminal = np.logical_or((dense_actions == eos_token_id), previous_node_is_terminal)
        # Evaluate nodes.
        (prior, values, next_states, classi_states) = self._rec_fun(
            mcts_object=self,
            original_states=original_states,
            classi_states=classi_states,
            original_token_ids=original_tokens_ids,
            original_attention_mask=original_attention_masks,
            target_labels=self._labels,
            likelihoods=likelihoods,
            parent_nodes=node_indices,
        )
        values[previous_node_is_terminal] = previous_values[previous_node_is_terminal]

        # Store unpaded version of inputs to save space
        original_attention_masks = [
            torch.cat((self._original_attention_mask[(b, n)], torch.ones(1, dtype=torch.long, device=self._lm_device)),
                      dim=0) for b, n in enumerate(node_indices)]
        original_tokens_ids = [
            torch.cat((self._original_token_ids[(b, n)], torch.tensor([dense_actions[b]], device=self._lm_device)),
                      dim=0)
            for b, n in enumerate(node_indices)]

        # Create the new nodes.
        self.create_node(next_node_index, prior, likelihoods * children_priors, values, next_states,
                         original_tokens_ids, original_attention_masks, classi_states, expanded_node_is_terminal)

        # Update the min and max values arrays
        # self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        # self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)

        # Update tree topology.
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        self._parents[:, next_node_index] = node_indices
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1

    def create_node(self, node_index, prior, likelihoods, values, original_states, original_tokens_ids,
                    original_attention_masks, classi_states, expanded_node_is_terminal):
        # Truncate the prior to only keep the top k logits
        if self._argpartition_reproducibility:
            kth = range(-self._num_sparse_actions, 0)
        else:
            kth = -self._num_sparse_actions
        prior_topk_indices = np.argpartition(prior, kth, axis=-1)[:, -self._num_sparse_actions:]
        prior = prior[self._batch_range[:, None], prior_topk_indices]  # (B, A)
        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices

        # Update prior, values and visit counts.
        self._children_prior[:, node_index, :] = prior
        self._likelihoods[:, node_index] = likelihoods
        # raw_values = values**(self.alpha) * likelihoods**(1-self.alpha)
        raw_values = values
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal
        # States has shape [12 (nlayer), 2(key/value), batch_size, 12(nhead), seq_len, 64]
        if self._classifier_caching:
            classi_key_value_tensor = torch.stack(
                list(torch.stack(list(classi_states[i]), dim=0) for i in range(len(classi_states))), dim=0)
        # If root, store the whole states
        if (node_index == 0):
            for b in range(len(original_tokens_ids)):
                if self._lm_caching:
                    self._original_states[(b, node_index)] = [
                        [torch.clone(original_states[layer][kv][b]) for kv in range(len(original_states[layer]))]
                        for layer in range(len(original_states))
                    ]
                if self._classifier_caching:
                    self._classi_states[(b, node_index)] = torch.clone(classi_key_value_tensor[:, :, b])
        # Else just store the additional token hidden states to save space
        else:
            for b in range(len(original_tokens_ids)):
                if self._lm_caching:
                    self._original_states[(b, node_index)] = [
                        [
                            # decoder [:2], consult `https://huggingface.co/docs/transformers/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast.past_key_values`
                            torch.clone(original_states[layer][kv][b, :, -1:])
                            for kv in range(len(original_states[layer]))[:2]
                        ]
                        for layer in range(len(original_states))
                    ]
                if self._classifier_caching:
                    self._classi_states[(b, node_index)] = torch.clone(classi_key_value_tensor[:, :, b, :, -1:])

        # Updates tokens ids
        for b, original_token_ids in enumerate(original_tokens_ids):
            self._original_token_ids[(b, node_index)] = original_token_ids

        # Updates attention masks
        for b, original_attention_mask in enumerate(original_attention_masks):
            self._original_attention_mask[(b, node_index)] = original_attention_mask

    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices  # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            if is_root.all():
                return
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            root_mask = 1.0 * is_root
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            self._values[self._batch_range, parents] = (
                    not_root_mask * (self._values[self._batch_range, parents]
                                     * self._visit_counts[self._batch_range, parents] + leaf_values)
                    / (self._visit_counts[self._batch_range, parents] + 1.0)
                    + root_mask * self._values[self._batch_range, parents]
            )
            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            self._children_values[self._batch_range, parents, actions] = (
                    not_root_mask * self._values[self._batch_range, node_indices]
                    + root_mask * self._children_values[self._batch_range, parents, actions]
            )
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            node_indices = parents

    def create_pygraphviz_agraph(self, tokenizer, batch_idx):
        G = self.create_nx_digraph(tokenizer, batch_idx)
        A = nx.nx_agraph.to_agraph(G)
        A.layout('dot', args='-Nfontsize=12 -Nwidth=".3" -Nheight=".3" -Nmargin=0 -Gfontsize=8')
        return A

    def create_nx_digraph(self, tokenizer, batch_idx):
        root_idx = 0
        G = nx.DiGraph()
        G.add_node(
            root_idx,
            sequence=tokenizer.decode(self._original_token_ids[batch_idx, 0]),
            **self._get_root_attributes(batch_idx, root_idx)
        )
        self._add_tree_to_nx_digraph(tokenizer, batch_idx, G, 0, 1.0)

        for node_idx in G.nodes:
            G.nodes[node_idx]["label"] = NumpyMCTS.attributes_to_str(idx=node_idx, **G.nodes[node_idx])
        for edge_idx in G.edges:
            G.edges[edge_idx]["label"] = NumpyMCTS.attributes_to_str(**G.edges[edge_idx])

        return G

    def _add_tree_to_nx_digraph(self, tokenizer, batch_idx, G, node_idx, likelihood):
        if node_idx == -1:
            return

        node_children_prior = self._children_prior[batch_idx, node_idx, :]
        node_children_values = self._children_values[batch_idx, node_idx, :]
        node_children_visits = self._children_visits[batch_idx, node_idx, :]
        node_visits = self._visit_counts[batch_idx, node_idx]
        node_policy_score = np.sqrt(node_visits) * self._pb_c_init * node_children_prior / (node_children_visits + 1)
        node_value_score = node_children_values

        for action in range(len(node_children_prior)):
            token_id = self._topk_mapping[batch_idx, node_idx, action]
            token = tokenizer.decode([token_id])
            child_idx = int(self._children_index[batch_idx, node_idx, action])
            is_child_leaf = child_idx == -1

            if is_child_leaf:
                child_key = (node_idx, action)
                G.add_node(child_key)
            else:
                child_key = child_idx
                G.add_node(
                    child_key,
                    sequence=tokenizer.decode(self._original_token_ids[batch_idx, child_idx]),
                    **self._get_node_attributes(batch_idx, child_idx)
                )

            G.add_edge(
                node_idx,
                child_key,
                action=int(action),
                token=token,
                token_id=int(token_id),
                p_action=float(self._children_prior[batch_idx, node_idx, action]),
                score=float(node_policy_score[action] + node_value_score[action]),
                policy_score=float(node_policy_score[action]),
                value_score=float(node_value_score[action]),
            )

            p_token_given_history = node_children_prior[action]
            p_token_and_history = likelihood * p_token_given_history
            self._add_tree_to_nx_digraph(tokenizer, batch_idx, G, child_idx, p_token_and_history)

    def _get_root_attributes(self, batch_idx, root_idx=0):
        attributes = {
            "batch_idx": int(batch_idx),
            "batch_size": int(self._batch_size),
            "num_simulations": int(self._num_simulations),
            "num_actions": int(self._num_actions),
            "num_sparse_actions": int(self._num_sparse_actions),
            "num_nodes": int(self._num_nodes),
            "target_label": float(self._labels[batch_idx, 0]),
            "pb_c_init": float(self._pb_c_init),
        }
        attributes.update(self._get_node_attributes(batch_idx, root_idx))
        return attributes

    def _get_node_attributes(self, batch_idx, node_idx):
        attributes = {
            "parent": int(self._parents[batch_idx, node_idx]),
            "depth": int(self._depth[batch_idx, node_idx]),
            "p_x": float(self._likelihoods[batch_idx, node_idx]),
            "avg_value": float(self._values[batch_idx, node_idx]),
            "raw_value": float(self._raw_values[batch_idx, node_idx]),
            "visits": int(self._visit_counts[batch_idx, node_idx]),
            "is_terminal": bool(self._is_terminal[batch_idx, node_idx]),
        }
        return attributes

    @staticmethod
    def attributes_to_str(**kwargs):
        desc = "\n".join([f"{key}: {value:.3f}"
                          if isinstance(value, float) or isinstance(value, np.float32)
                          else f"{key}: {value}"
                          for key, value in kwargs.items()])
        desc = desc.strip('\n')
        return desc
