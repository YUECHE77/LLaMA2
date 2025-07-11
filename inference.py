import argparse
import torch
from llama.utils import setup_seeds, load_model_and_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the LLaMA2 model",
                        default="/data3/yueche/Llama-2-7b-chat")
    parser.add_argument("--max-len", type=int, help="The max generation length", default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    setup_seeds(seed=727)

    model, tokenizer = load_model_and_tokenizer(args.model_path, torch_dtype=torch.float16)
    model.eval()

    prompts = [
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    results = model.generate(
        tokenizer, 
        prompt_tokens, 
        max_gen_len=args.max_len, 
        temperature=args.temperature, 
        top_p=args.top_p,
    )

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
