import argparse
import copy
import json
import logging
import time
from tqdm import tqdm
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from llama.utils import setup_seeds, load_model_and_tokenizer, enable_lora

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def _tokenize_fn(strings, tokenizer):
    tokenized_list = [torch.tensor(tokenizer.encode(text, bos=True, eos=True)) for text in strings]
    input_ids = labels = [tokenized for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.ne(tokenizer.pad_id).sum().item() for tokenized in tokenized_list]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len-1] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path, tokenizer):
        super(SupervisedDataset, self).__init__()

        print("Loading data...")
        with open(data_path, "r") as f:
            list_data_dict = json.load(f)

        print("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}" for example in list_data_dict]

        print("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    def __call__(self, instances: List[Dict[str, torch.Tensor]]):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))  # Both: List[torch.Tensor]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(tokenizer.pad_id),
        )


def make_supervised_data_module(tokenizer, data_path):
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_path)
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the LLaMA2 model",
                        default="/data3/yueche/Llama-2-7b-chat")
    parser.add_argument("--data-path", type=str, help="Path to the training data",
                        default="/home/yueche/miniconda3/LLaMA2/alpaca_data_200_samples.json")
    parser.add_argument("--save-path", type=str, help="The path to save the lora weights",
                        default="/data3/yueche/myLLaMA2_lora/lora_weights.pth")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-5)
    parser.add_argument("--accumulate-steps", type=int, help="Gradient accumulate steps", default=8)
    args = parser.parse_args()
    setup_seeds(727)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    tokenizer.pad_id = tokenizer.unk_id  # Avoid negative padding

    # Freeze all layers except the LoRA layers
    enable_lora(model)

    # Create dataloader
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_path=args.data_path)
    dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=args.batch_size,
        collate_fn=data_module["data_collator"],
        shuffle=True,
    )

    # Prepare optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    model.train()
    scaler = torch.amp.GradScaler('cuda')  # Automatic Mixed Precision Training
    iters_to_accumulate = args.accumulate_steps

    start = time.time()
    for epoch in range(5):
        print(f"Epoch {epoch + 1}")
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            input_ids = batch['input_ids'].to("cuda")  # [B, seq_len]
            labels = batch['labels'].to("cuda")  # [B, seq_len]

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids, start_pos=0)  # [B, seq_len, vocab_size]
                shift_logits = logits[..., :-1, :].contiguous()  # [B, seq_len - 1, vocab_size]
                shift_labels = labels[..., 1:].contiguous()  # [B, seq_len - 1]
                shift_logits = shift_logits.view(-1, tokenizer.n_words)  # vocab_size = tokenizer.n_words = 32000
                shift_labels = shift_labels.view(-1)

                loss = criterion(shift_logits, shift_labels) / iters_to_accumulate

            scaler.scale(loss).backward()
            if (i + 1) % iters_to_accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if (i + 1) % 50 == 0:
                print(f"Loss: {loss.item()}")

    end = time.time()
    print(f"Training Time: {end - start}")

    # Save LoRA weights
    model_weights = model.state_dict()
    lora_weights = {k: v for k, v in model_weights.items() if "lora_" in k}
    torch.save(lora_weights, args.save_path)
    print(f'Lora weights (after training) has been saved to: {args.save_path}')
