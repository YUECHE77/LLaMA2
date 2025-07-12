import argparse
import time
import torch
from llama.utils import setup_seeds, load_model_and_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="Path to the LLaMA2 model",
                        default="/data3/yueche/Llama-2-7b-chat")
    parser.add_argument("--lora-path", type=str, help="If you have finetuned", default=None)
    parser.add_argument("--max-len", type=int, help="The max generation length", default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--report-time", action="store_true", help="Report the inference time")
    args = parser.parse_args()
    setup_seeds(seed=727)

    model, tokenizer = load_model_and_tokenizer(
        model_dir=args.model_path, 
        lora_path=args.lora_path, 
        torch_dtype=torch.float16,
    )
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
    # prompts = [
    #     "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nWhat are the three primary colors?\n\n### Response:"
    # ]
    prompt_tokens = [tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    assert len(prompt_tokens) <= model.params.max_batch_size, f'Exceed maximum batch_size ({model.params.max_batch_size})'

    start = time.time()
    results = model.generate(
        tokenizer, 
        prompt_tokens, 
        max_gen_len=args.max_len, 
        temperature=args.temperature, 
        top_p=args.top_p,
        use_cache=True,  # it must be True
    )
    end = time.time()

    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")
    
    if args.report_time:
        print(f'\nInference time: {end - start}')
