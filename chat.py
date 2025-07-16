import argparse
from datetime import datetime

import torch
from llama.utils import setup_seeds, load_model_and_tokenizer

CURRENT = datetime.now().strftime("%d %b %Y")
SYSTEM = f'You are LLaMA2, created by Meta. Today Date: {CURRENT}. You are a helpful assistant.'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the LLaMA2 model",
                        default="/data3/yueche/Llama-2-7b-chat")
    parser.add_argument("--lora-path", type=str, help="If you have finetuned", default=None)
    parser.add_argument("--max-len", type=int, help="The max generation length", default=128)
    parser.add_argument("--sampling", action="store_true", help="Use Nucleus sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    setup_seeds(seed=727)

    model, tokenizer = load_model_and_tokenizer(
        model_dir=args.model_path, 
        lora_path=args.lora_path, 
        torch_dtype=torch.float16,
    )
    model.eval()

    dialog = [
        {'role': 'system', 'content': SYSTEM},
    ]
    while True:
        query = input('User: ')
        if query.strip().lower() in ['exit', 'leave', 'end', 'break', 'goodbye', 'bye']: break
        dialog.append({'role': 'user', 'content': query})

        response = model.chat(
            tokenizer,
            [dialog, ],
            max_gen_len=args.max_len, 
            sampling=args.sampling,
            temperature=args.temperature, 
            top_k=args.top_k, 
            top_p=args.top_p,
            use_cache=True,  # it must be True
        )[0]['generation']

        print('Assistant: ' + response['content'])
        dialog.append(response)
