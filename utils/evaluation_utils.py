import re
from typing import Union

import torch
import torch.nn as nn
from datasets import load_dataset

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.quantization_utils import int_activation_quantization_pre_hook


@torch.no_grad()
def evaluate_model(model: nn.Module,
                   tokenizer: AutoTokenizer,
                   tasks: str,
                   eval_ppl: str,
                   num_fewshot: int = 0,
                   limit: Union[int, float] = -1,
                   batch_size: int = 1):
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device_map="auto")
    print(f"[DEBUG] num_fewshot: {num_fewshot}")
    lm.seqlen = 2048
    ppl_results = {}
    task_results = {}
    if eval_ppl:
        for dataset in eval_ppl.split(","):
            if "c4" in dataset:
                testdata = load_dataset(
                    dataset,
                    "en",
                    split="validation",
                )
                testloader = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
                testenc = testloader.input_ids[:, :(256 * lm.seqlen)]
            else:
                testdata = load_dataset(
                    dataset,
                    split="test",
                )
                testloader = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
                testenc = testloader.input_ids
            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)].to(lm.device)
                outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen): ((i + 1) * lm.seqlen)][:, 1:].to(shift_logits.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            ppl_results[dataset] = ppl.item()
    if tasks != "":
        csr_results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
        )

        csr_results = csr_results["results"]
        task_results.update(csr_results)
        print(task_results)

    clean_results = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in task_results.items()}

    return ppl_results, task_results, clean_results


def evaluation(model: nn.Module,
               model_type: str,
               tokenizer: AutoTokenizer,
               tasks: str,
               ppls: str,
               quantization_bit: int,
               input_group_size: int,
               is_baseline: bool = False):
    if not is_baseline:
        if model_type == "qwen":
            pattern = r".*(q_proj|k_proj|v_proj|gate|gate_proj|up_proj).*"
        elif model_type == "mixtral":
            pattern = r".*(q_proj|k_proj|v_proj|gate|w1|w3).*"
        elif model_type == "deepseek":
            pattern = r".*(q_proj|kv_a_proj_with_mqa|gate|gate_proj|up_proj).*"
        regex = re.compile(pattern)
        
        for name, module in model.named_modules():
            if regex.search(name):
                hook = int_activation_quantization_pre_hook(module_name=name,
                                                        quantization_bit=quantization_bit,
                                                        input_group_size=input_group_size,
                                                        output_dict=None)
                module.register_forward_pre_hook(hook)

    ppl_results, task_results, clean_results = evaluate_model(model,
                                                              tokenizer,
                                                              tasks=tasks,
                                                              eval_ppl=ppls,
                                                              batch_size=8)

    return ppl_results, task_results, clean_results

