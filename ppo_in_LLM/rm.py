import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pylab as plt
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import Dataset, load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

args = {
    "seed": 42,
    'model_name_or_path': 'facebook/opt-350m', # using a small model compared to the GPT for reward training
    'learning_rate': 5e-5,
    'batch_size': 4,
    'gradient_accumulation_steps': 16,
    'num_train_epochs': 1,
    'num_workers': 10,
    'seq_length': 1024,
    'logging_steps': 10,
}

args = DictConfig(args)

set_seed(args.seed)

tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# imdb dataset for sentiment analysis
raw_dataset = load_dataset("imdb")
del raw_dataset['unsupervised']

def create_custom_dataset(raw_dataset):
    df = raw_dataset.to_pandas()
    negative_df = df[df['label']==0]
    positive_df = df[df['label']==1]
    negative_df = negative_df.drop(
        columns=['label']).rename(
        columns={'text': 'rejected'})
    # shuffle the data
    positive_df = positive_df.sample(
        frac=1, random_state=0).reset_index(
        drop=True).drop(columns=['label']).rename(
        columns={'text': 'chosen'})
    joined_df = negative_df.join(positive_df)

    def tokenize_fn(texts, max_length=args.seq_length):
        encoded = tokenizer(
            texts,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return encoded
    # the tokenizer returns a dict with `input_ids` being the key of one list and `attention_mask` being the key of another list
    rejected_encoded = tokenize_fn(joined_df.rejected.values.tolist())
    # add as a new column in joined_df
    joined_df['rejected_input_ids'] = rejected_encoded['input_ids']
    joined_df['rejected_attention_mask'] = rejected_encoded['attention_mask']
    encoded_chosen = tokenize_fn(joined_df.chosen.values.tolist())
    # add as a new column in joined_df
    joined_df['chosen_input_ids'] = encoded_chosen['input_ids']
    joined_df['chosen_attention_mask'] = encoded_chosen['attention_mask']

    train_dataset = Dataset.from_pandas(joined_df, preserve_index=False)
    # It changes the format of the dataset so that when you access a sample (e.g. train_dataset[0]),
    # the returned data is in PyTorch tensor format instead of regular Python types like lists or numpy arrays.
    return train_dataset.with_format("torch")

# 12500 pairs
train_dataset = create_custom_dataset(raw_dataset['train'])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)


# causal LM reward head
# add a new head on top of the base model by taking its last hideen size
class CLMRewardHead(nn.Module):
    def __init__(self, base_model):
        super(CLMRewardHead, self).__init__()
        self.base_model = base_model
        self.device = self.base_model.device
        # n_embd = self.base_model.config.hidden_size
        n_embd = 512 # self.base_model.device = 1024, doesn't match the hidden output
        self.reward_head = nn.Sequential(
                nn.LayerNorm(n_embd),
                nn.GELU(),
                nn.Linear(n_embd, 4*n_embd),
                nn.GELU(),
                nn.Linear(4*n_embd, 1),
            ).to(torch.bfloat16).to(self.device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1] # (batch_size, seq_length, hidden_dim=512)
        last_hidden_state_mean = torch.mean(last_hidden_state, 1) # use the average of hidden states across the whole sequence to represent the sequence
        last_hidden_state_mean = last_hidden_state_mean.to(torch.bfloat16)
        logits = self.reward_head(last_hidden_state_mean)
        return logits

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto", # use device_map='auto' argument to allow your model to be distributed over your devices (GPUs/TPUs)
)

model = CLMRewardHead(base_model)

def train():
    epoch = 1
    print_interval=args.logging_steps
    num_batches = len(train_dataloader)
    progress_bar = tqdm(total=num_batches*args.num_train_epochs, leave=True)
    progress_bar.set_description(f"| Train: Epoch {epoch}, evaluating ... |")
    losses = []
    temp_losses = []
    i = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            print(i)
            chosen_input_ids = batch['chosen_input_ids'].to(model.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(model.device)
            rejected_input_ids = batch['rejected_input_ids'].to(model.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(model.device)

            r_w = model(chosen_input_ids, attention_mask=chosen_attention_mask)
            r_l = model(rejected_input_ids, attention_mask=rejected_attention_mask)
            loss = -F.logsigmoid(r_w - r_l).mean()

            # Accumulate the gradients
            loss /= args.gradient_accumulation_steps
            # It computes the gradients of the loss with respect to all model parameters.
            # It adds these gradients to the existing .grad fields (instead of overwriting them) to accumulate gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if (i + 1) % args.gradient_accumulation_steps == 0 or i + 1 == len(train_dataloader):
                optimizer.step()
                # clear the previous gradients after we update params
                optimizer.zero_grad()

            temp_losses.append( loss.item() )
            if i%print_interval==0:
                progress_bar.set_description(f"| Train: Epoch {epoch}, loss = {loss.item():4f} |")
                progress_bar.refresh()
                losses.append( np.mean(temp_losses) )
                temp_losses = []
            progress_bar.update()
            i+=1

    progress_bar.set_description(f"| Train: Epoch {epoch}, loss = {loss.item():4f} |")
    progress_bar.refresh()

def save():
    torch.save(model.state_dict(), "fb_opt_350m_rm.pth")

def load():
    model = CLMRewardHead(base_model)
    model.load_state_dict(torch.load('fb_opt_350m_rm.pth'))

def eval():
    model = model.eval()
    test_dataset = create_custom_dataset(raw_dataset['test'])
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    num_correct_orders = 0
    with torch.no_grad():

        for batch in tqdm(test_dataloader):

            chosen_input_ids = batch['chosen_input_ids'].to(model.device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(model.device)
            rejected_input_ids = batch['rejected_input_ids'].to(model.device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(model.device)

            r_w = model(chosen_input_ids, attention_mask=chosen_attention_mask).logits
            r_l = model(rejected_input_ids, attention_mask=rejected_attention_mask).logits

            num_correct_orders += (r_w - r_l>0).sum().item()

    print('Accuracy of orders after training: ', num_correct_orders/(len(test_dataloader)*args.batch_size))


def main():
    train()
    save()
    eval()

if __name__ == "__main__":
    main()