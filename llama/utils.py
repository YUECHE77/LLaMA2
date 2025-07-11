import os
import glob
import random
import numpy as np
from typing import Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from .tokenizer import Tokenizer
from .model import ModelArgs, Llama

def setup_seeds(seed: int = 727):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def load_model_and_tokenizer(
    model_dir: str,
    lora_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    device: str = "cuda",
) -> Tuple[Llama, Tokenizer]:
    """
    Load base model (plus optional LoRA) and the SentencePiece tokenizer.

    Args:
        model_dir (`str`): Folder that contains *.pth weights and *.model tokenizer
        lora_path (`str`): Path to the saved LoRA state_dict
        torch_dtype (`torch.dtype`): torch.float16 / bfloat16 / float32, etc.
        device (`str`): "cuda" / "cpu"
    """
    spm_files = glob.glob(os.path.join(model_dir, "*.model"))
    if not spm_files:
        raise FileNotFoundError("No *.model tokenizer file found in %s" % model_dir)
    tokenizer_path = spm_files[0]
    tokenizer = Tokenizer(tokenizer_path)

    ckpt_files = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    if not ckpt_files:
        raise FileNotFoundError("No *.pth checkpoint found in %s" % model_dir)
    if len(ckpt_files) > 1:
        print(f"[load_model] Found multiple *.pth files, picking {ckpt_files[0]}")
    checkpoint = torch.load(ckpt_files[0], map_location='cpu', weights_only=True)

    model_args = ModelArgs()
    if torch_dtype is not None:
        model = Llama(model_args).to(torch_dtype)
    else:
        model = Llama(model_args)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    assert not missing, f"Missing Base model weights: {missing}"

    if lora_path is not None:
        lora_weights = torch.load(lora_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(lora_weights, strict=False)
        assert not missing, f"Missing LoRA weights: {missing}"

    model.to(device)

    return model, tokenizer

def enable_lora(model: Llama):
    for name, param in model.named_parameters():
        if 'lora_' in name:
            # Names are: layers.{i}.attention.wq.lora_A / layers.{i}.attention.wq.lora_B
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable params: {trainable_params:,d} || "
        f"All params: {all_params:,d} || "
        f"Trainable%: {100 * trainable_params / all_params:.2f}"
    )

def merge_lora(lora_path: str, model_base: str, save_folder: str):
    """
    Merge the lora weights with the original model weights, and then 
    save the merged model. You can also just provide the lora_path when
    initialize the model using load_model_and_tokenizer(). So that the
    weights are automatically merged when "model.eval()".

    Args:
        lora_path (`str`): Path to the lora weights (after training -> the .pth file).
        model_base (`str`): Path to the FOLDER of the base model (LLaMA2 folder).
        save_folder  (`str`): Folder to save the merged model and tokenizer.
    """
    os.makedirs(save_folder, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(
        model_base,
        lora_path=lora_path,
        device='cpu',
    )
    model.eval()

    merge_path = os.path.join(save_folder, "merged_model.pth")
    torch.save(model.state_dict(), merge_path)
    print(f"[merge_lora] merged weights saved to {merge_path}")
