import torch
from torch import nn

from typing import Optional, List, Dict, Literal, TypedDict

from llama.tokenizer import Tokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str

class Generation(nn.Module):
    def generate(
        self, 
        tokenizer: Tokenizer, 
        prompt_tokens: List[List[int]], 
        max_gen_len: int = 128, 
        sampling: bool = False,
        temperature: Optional[float] = None, 
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        echo: bool = False,  # Flag indicating whether to include prompt tokens in the generated output.
        use_cache: bool = True,
    ) -> List[List[int]]:
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
            
            if temperature is not None:
                assert isinstance(temperature, int) or isinstance(temperature, float), (
                    f'temperature should be a number (either int or float), but got temperature = {temperature}.'
                )
                if temperature > 0 and sampling:
                    probs = torch.softmax(logits[:, -1] / temperature, dim=-1)  # [B, vocab_size]

            if sampling:
                next_token = self.sample_top_p(probs, top_k, top_p)  # [B, 1]
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

        return out_tokens


    def sample_top_p(
        self, 
        probs: torch.Tensor, 
        top_k: Optional[int] = None, 
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # Both: [B, vocab_size]

        if top_k is not None:
            assert top_k > 0 and top_k < probs_sort.shape[-1], (
                f'top_k should be: 0 < top_k < vocab_size ({probs_sort.shape[-1]}), '
                f'but got top_k = {top_k}.'
            )
            probs_sort[:, top_k:] = 0.0

        if top_p is not None:
            assert 0.0 < top_p < 1.0, (
                f'top_p should be: 0.0 < top_p < 1.0, '
                f'but got top_p = {top_p}.'
            )
            probs_sum = torch.cumsum(probs_sort, dim=-1)  # [B, vocab_size]
            p_mask = probs_sum - probs_sort > top_p
            probs_sort[p_mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True) + 1e-12)  # element-wise division
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token

    def chat(
        self, 
        tokenizer: Tokenizer, 
        dialogs: List[List[Message]],
        max_gen_len: int = 128, 
        sampling: bool = False,
        temperature: Optional[float] = None, 
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_cache: bool = True,
    ):
        max_seq_len = self.params.max_seq_len  # default: 1024
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )

            # Combine the system prompt with the first user message
            if dialog[0]['role'] == 'system':
                combined_role = dialog[1]['role']
                combined_content = B_SYS + dialog[0]['content'] + E_SYS + dialog[1]['content']
                dialog = [
                    {
                        "role": combined_role,
                        "content": combined_content,
                    }
                ] + dialog[2:]

            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            dialog_tokens: List[int] = sum(
                [
                    tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )

            assert dialog[-1]["role"] == "user", f"Last message must be from user, got {dialog[-1]['role']}"

            dialog_tokens += tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )

            prompt_tokens.append(dialog_tokens)
        
        out_tokens = self.generate(
            tokenizer,
            prompt_tokens,
            max_gen_len=max_gen_len,
            sampling=sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            echo=False,  # Must be False
            use_cache=use_cache,
        )

        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(out_tokens, unsafe_requests)
        ]