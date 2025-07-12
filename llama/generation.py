import torch
from torch import nn

from typing import Optional, List, Dict

from llama.tokenizer import Tokenizer

class Generation(nn.Module):
    def generate(
        self, 
        tokenizer: Tokenizer, 
        prompt_tokens: List[List[int]], 
        max_gen_len: int = 128, 
        temperature: float = 0.7, 
        top_p: float = 0.9,
        echo: bool = False,
        use_cache: bool = True,
    ) -> List[Dict[str, str]]:
        bsz = len(prompt_tokens)
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_len = max_gen_len + max_prompt_len

        tokens = torch.full((bsz, total_len), tokenizer.pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != tokenizer.pad_id

        for cur_pos in range(min_prompt_len, total_len):
            with torch.no_grad():
                logits = self(tokens[:, prev_pos:cur_pos], prev_pos, use_cache)  # [B, q_len, vocab_size] where q_len = cur_pos - prev_pos

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)  # [B, vocab_size]
                next_token = self.sample_top_p(probs, top_p)  # [B, 1]
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)  # [B, ]

            next_token = next_token.reshape(-1)  # [B, ]
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == tokenizer.eos_id
            )  # "|=" -> bitwise OR assignment

            prev_pos = cur_pos

            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max_gen_len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]

            # cut to eos tok if any
            if tokenizer.eos_id in toks:
                eos_idx = toks.index(tokenizer.eos_id)
                toks = toks[:eos_idx]

            out_tokens.append(toks)

        return [{"generation": tokenizer.decode(t)} for t in out_tokens]


    def sample_top_p(self, probs: torch.Tensor, p: float):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # Both: [B, vocab_size]
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token
