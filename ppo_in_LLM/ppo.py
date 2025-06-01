import random
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pylab as plt
from omegaconf import DictConfig
from dataclasses import dataclass
from torchtyping import TensorType
from typing import Iterable, Sequence, List, Tuple

from rm import CLMRewardHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)

set_seed(2023)

args = {
    'generator_model_name_or_path': 'facebook/opt-1.3b',
    'reward_model_path': 'fb_opt_350m_rm.pth', # the reward model trained in rm.py
    'reward_model_base': 'facebook/opt-350m',
    'seq_length': 1024,
    'batch_size': 64,
    'lr': 0.00006,
    'prompt_size': 30,
    'prompt_batch_size': 128,
    'num_rollouts': 128,
    'epochs': 100,
    'ppo_epochs': 4,
    'gen_kwargs': {
        'max_new_tokens': 40,
        'top_k': 0,
        'top_p': 1.0,
        'do_sample': True
    },
    'kl_coef': 0.01,
    'gamma': 1,
    'lam': 0.95,
    'cliprange': 0.2,
    'cliprange_value': 0.2,
    'vf_coef': 1,
}

args = DictConfig(args)

tokenizer = AutoTokenizer.from_pretrained(args.generator_model_name_or_path)

class PromptDataset():
    def __init__(self, prompt_size, data):
        all_input_ids = tokenizer([item for item in data['train']['text']]).input_ids
        # note here: every input prompt is a truncated version
        self.prompts_input_ids = np.array([item[:prompt_size] for item in all_input_ids if len(item)>prompt_size])

    def __getitem__(self, ix):
        return self.prompts_input_ids[ix]

    def __len__(self):
        return len(self.prompts_input_ids)


# iterator for the training dataset
# used to create rollout
class CustomPromptDataGenerator():
    def __init__(self, prompt_dataset, prompt_batch_size):
        self.prompt_dataset = prompt_dataset
        self.prompt_batch_size = prompt_batch_size

    def __iter__(self):
        self.dataset_indices = np.arange(len(self.prompt_dataset))
        return self

    def __next__(self):
        if len(self.dataset_indices) >= self.prompt_batch_size:
            picked_indices = np.random.choice(np.arange(len(self.dataset_indices)),
                                              self.prompt_batch_size,
                                              replace=False)
            samples = self.prompt_dataset[self.dataset_indices[picked_indices]]
            self.dataset_indices = np.delete(self.dataset_indices, picked_indices)
            input_ids = torch.tensor(samples)
            # since every input is a truncated version, the attention mask is all 1s
            attention_mask = torch.ones_like(input_ids)
            batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
            return batch
        else:
            raise StopIteration

# use symbolic dimention names for type checking during runtime
@dataclass
class PPORLElement:
    query_tensor: TensorType["query_size"] # prompt length
    response_tensor: TensorType["response_size"] # sequence length
    logprobs: TensorType["response_size", "vocab_size"]
    values: TensorType["response_size"]
    rewards: TensorType["response_size"]


@dataclass
class PPORLBatch:
    query_tensors: TensorType["batch_size", "query_size"]
    response_tensors: TensorType["batch_size", "response_size"]
    logprobs: TensorType["batch_size", "response_size", "vocab_size"]
    values: TensorType["batch_size", "response_size"]
    rewards: TensorType["batch_size", "response_size"]


# convert rollout to dataloader for training
class PPORolloutStorage():

    def __init__(self):
        super().__init__()
        self.pad_token_id = tokenizer.pad_token_id
        self.history: Iterable[PPORLElement] = [None]

    def push(self, exps: Iterable[PPORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    # let object behave like a list which is useful for dataloader
    def __getitem__(self, index: int) -> PPORLElement:
        return self.history[index]
    
    # let object behave like a list which is useful for dataloader
    def __len__(self) -> int:
        return len(self.history)

    def create_loader(self, mini_batch_size: int, shuffle: bool) -> DataLoader:
        # combine a list of data samples into a single batch
        # since prompts and responses often have variable lengths, need to pad (pad_sequnce) before batching
        def collate_fn(elems: Iterable[PPORLElement]):
            return PPORLBatch(
                pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.values for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True
                ),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
            )

        return DataLoader(self, mini_batch_size, shuffle=shuffle, collate_fn=collate_fn)


class Agent(nn.Module):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.model = AutoModelForCausalLM.from_pretrained(
            args.generator_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        if not self.trainable:
            self.model = self.model.eval()
            self.model.requires_grad_(False)
        else:
            n_embd = self.model.lm_head.in_features
            num_labels = 1
            self.value_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, num_labels),
            ).to(torch.bfloat16).to(self.model.device)
        # get_output_embeddings() returns the final linear layer mapping hidden states to vocabulary logits
        # it's equivalent to Linear(in_features=output_hidden_dim, out_features=vocab_size)
        self.logit_head = self.model.get_output_embeddings()

    # **x captures extra keyword arguments as a dictionary accepted by transformers.AutoModelForCausalLM.generate
    def generate(self, input_ids, **x):
        return self.model.generate(input_ids, **x)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids,
                             attention_mask=attention_mask,
                             output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # (batch_size, seq_len, hidden_dim)
        lm_logits = self.logit_head(last_hidden_state) # (batch_size, seq_len, vocab_size)
        if self.trainable:
            value = self.value_head(last_hidden_state).squeeze(-1) # (batch_size, seq_len)
            return lm_logits, value
        else:
            return lm_logits


base_model = AutoModelForCausalLM.from_pretrained(
    args.reward_model_base,
    torch_dtype=torch.bfloat16,
    device_map="auto", # use device_map='auto' argument to allow your model to be distributed over your devices (GPUs/TPUs)
)
reward_model = CLMRewardHead(base_model)
# load trained RM model from rm.py
reward_model.load_state_dict(torch.load(args.reward_model_path))
reward_model.eval() # set to eval model for inference only


def reward_fn(samples):
    ins = tokenizer(samples, padding=True, truncation=True, max_length=args.seq_length, return_tensors='pt')
    logits = reward_model(**ins.to(reward_model.device)) # (batch_size, 1) given reward model only output one value per sample
    temperature = 0.3 # Common in reward modeling to stabilize reward distribution
    # [:, 0] is necessary if we want a flat list of values instead of a list of singleton lists
    sentiments = torch.sigmoid(logits*temperature)[:,0].detach().cpu().tolist() # (batch_size)
    return sentiments


def logprobs_from_logits(logits, labels):
    '''
    logits: (batch_size, seq_length, vocab_size)
    labels: (batch_size, seq_length)
    note: labels are ids indicating the true next prediction
    '''
    # (batch_size, seq_length, vocab_size)
    logprobs = F.log_softmax(logits, dim=-1)
    # (batch_size, seq_length), extracts the log-prob of the correct token at each timestep.
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)) 
    return logprobs_labels.squeeze(-1)


class RolloutCreator():
    def __init__(
            self,
            prompt_dataset,
            prompt_batch_size=args.prompt_batch_size,
    ):
        self.prompt_batch_size = prompt_batch_size
        self.prompt_dataset = prompt_dataset
        self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
        self.prompt_iterator = iter(self.prompt_generator)
        self.generate_kwargs = dict(
            args.gen_kwargs,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )


    def make_experience(self, model, num_rollouts=128):

        all_rollouts = []
        while len(all_rollouts) < num_rollouts:
            try:
                batch = next(self.prompt_iterator)
            except StopIteration:
                # restart from beginning if data exhausted
                self.prompt_generator = CustomPromptDataGenerator(self.prompt_dataset, self.prompt_batch_size)
                self.prompt_iterator = iter(self.prompt_generator)
                batch = next(self.prompt_iterator)

            query_tensors = batch['input_ids'].to(model.model.device)
            trajectories = model.generate(
                query_tensors,
                attention_mask=batch['attention_mask'].to(model.model.device),
                **self.generate_kwargs
            )
            # take only the generated one by taking out the initial prompt
            response_tensors = trajectories[:, query_tensors.shape[1]:]
            attention_mask = trajectories.not_equal(tokenizer.pad_token_id).long()

            with torch.no_grad():
                # logits: (batch_size, seq_len, vocab_size)
                # values: (batch_size, seq_len)
                logits, values = model(
                    trajectories,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_model(
                    trajectories,
                    attention_mask=attention_mask,
                )
            # teacher forcing as we predict the next token, true label is trajectories[batch_size, 1:]
            # where trajectories is [prompt] + [generated_content]
            # so this logprobs included the prompt part, that's why later we need start and ends to select parts for the loss computation
            logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
            ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], trajectories[:, 1:])
            n_trajectories = trajectories.shape[0]
            # values should take the same length as logits
            values = values[:, :-1]
            # start with the last prompt token as its output would be the first generated token
            # here we assume every element in the batch has the same prompt length
            # we can achieve this through truncation or padding
            start = batch['input_ids'].shape[1] - 1
            # ends with the place where attention_mask becomes 0
            ends = start + attention_mask[:, start:].sum(1)
            # pick values in the interval of [start, end]
            truncated_values = [values[i, start : ends[i]] for i in range(n_trajectories)]
            # pick logprobs in the interval of [start, end]
            truncated_logprobs = [logprobs[i, start : ends[i]] for i in range(n_trajectories)]

            texts = tokenizer.batch_decode(trajectories, skip_special_tokens=True)
            # not efficient as we decode and then encode in computing reward
            scores = reward_fn(texts) # (batch_size)
            # per-token KL penalty. Note this is not the full KL divergence over the whole vocabulary
            # Instead, it's just the log-prob of the action taken. So we don't need to include probability term.
            # idea is to make log prob as close to ref log prob as possible. The closer they are, the higher reward we should give. That's why we multiply -1 in front.
            # (batch_size, seq_length)
            rewards = -args.kl_coef * (logprobs - ref_logprobs)
            all_rewards = [None] * n_trajectories
            for i in range(n_trajectories):
                rs = rewards[i][start : ends[i]]
                # reward in the last token is the reward from the RM
                rs[-1] = scores[i]
                all_rewards[i] = rs

            new_rollout = [
                PPORLElement(
                    query_tensor=query_tensors[i],
                    response_tensor=response_tensors[i],
                    logprobs=truncated_logprobs[i],
                    values=truncated_values[i],
                    rewards=all_rewards[i],
                )
                for i in range(n_trajectories)
            ]
            all_rollouts += new_rollout

        score = torch.tensor(scores).mean().detach().cpu().item()

        return all_rollouts, score
    

def gae(
    values,
    rewards,
):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    # ideally we should let LLM estimate the value at t+1. However if it's in the terminal stage, setting to 0 is valid.
    last_value = 0
    with torch.no_grad():
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + args.gamma * last_value - values[:, t]
            last_advantage = delta + args.gamma * args.lam * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]
        returns = advantages + values
    return advantages, returns


def ppo_loss(
    logprobs,
    values,
    old_logprobs,
    old_values,
    advantages,
    returns,
    mask,
):
    values_clipped = torch.clamp(
        values,
        old_values - args.cliprange_value,
        old_values + args.cliprange_value,
    )
    n = mask.sum()
    # loss on value head
    # values_clipped is not necessary, we can use vf_loss1 directly
    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n
    # policy loss
    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n # average on valid tokens only
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n # average on valid tokens only
    loss = pg_loss + args.vf_coef * vf_loss
    return loss


def loss_fn(mini_batch):
    # batch from rollout
    query_tensors = mini_batch.query_tensors
    response_tensors = mini_batch.response_tensors
    old_logprobs = mini_batch.logprobs
    old_values = mini_batch.values
    old_rewards = mini_batch.rewards

    response_length = old_rewards.shape[1]

    advantages, returns = gae(old_values, old_rewards)

    trajectories = torch.hstack([mini_batch.query_tensors, mini_batch.response_tensors])
    attention_mask = trajectories.not_equal(tokenizer.pad_token_id).long()
    logits, values_pred = model(trajectories, attention_mask=attention_mask)
    # same as rollout which are generated from old model
    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[:, :-1, :], trajectories[:, 1:])
    attention_mask = attention_mask[:, :-1]

    start = query_tensors.shape[1] - 1
    end = start + response_length
    logprobs, values_pred, mask = (
        logprobs[:, start:end],
        values_pred[:, start:end],
        attention_mask[:, start:end],
    )

    loss = ppo_loss(
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        mask=mask,
    )

    return loss, old_rewards[:,-1].mean().item()


ref_model = Agent(trainable=False)
model = Agent(trainable=True)


# check output before ppo training
ins = tokenizer(
    ["This is an action Western.", "I saw this movie recently because"],
    return_tensors='pt',
    padding=True)

with torch.no_grad():
    outs = model.generate(
        ins['input_ids'].to(ref_model.model.device),
        attention_mask=ins['attention_mask'].to(ref_model.model.device),
        max_new_tokens=128,
        do_sample=True,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=2.,
    )

for i in range(len(outs)):
    generated_text = tokenizer.decode(outs[i], skip_special_tokens=True)
    print("\n" + "\033[1;30m" + generated_text)
    print("\033[1;32m" +'Score: ', np.round(reward_fn([generated_text]), decimals=4)[0], "\n")



imdb = load_dataset("imdb")
prompt_dataset = PromptDataset(args.prompt_size, imdb)

store = PPORolloutStorage()
rollout_creator = RolloutCreator(prompt_dataset, prompt_batch_size=args.prompt_batch_size)

opt = torch.optim.AdamW(model.parameters(), args.lr)

total_steps = (args.num_rollouts//args.batch_size)*args.ppo_epochs*args.epochs
tbar = tqdm(initial=0, total=total_steps)
all_scores = []

for i in range(args.epochs):
    # filling in the storage (phase 1)
    store.clear_history()
    rollouts, score = rollout_creator.make_experience(model, args.num_rollouts)
    store.push(rollouts)
    train_dataloader = store.create_loader(args.batch_size, shuffle=True)

    # loss calculation and graident optimization (Phase 2)
    for batch in train_dataloader:
        for _ in range(args.ppo_epochs):
            loss, reward = loss_fn(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            tbar.update()
    all_scores.append(score)
    tbar.set_description(f"| score: {score:.3f} |")

plt.plot(all_scores)


# inference after ppo training
ins = tokenizer(
    ["This is an action Western.", "I saw this movie recently because"],
    return_tensors='pt',
    padding=True
)

with torch.no_grad():
    outs = model.generate(
        ins['input_ids'].to(ref_model.model.device),
        attention_mask=ins['attention_mask'].to(ref_model.model.device),
        max_new_tokens=128,
        do_sample=True,
        temperature=0.5,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=2.,
    )

for i in range(len(outs)):
    generated_text = tokenizer.decode(outs[i], skip_special_tokens=True)
    print("\n" + "\033[1;30m" + generated_text)
    print("\033[1;32m" +'Score: ', np.round(reward_fn([generated_text]), decimals=4)[0], "\n")