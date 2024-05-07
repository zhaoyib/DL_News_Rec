import torch
from model.NRMS import NRMS
from model.quickstart import get_NRMS
from data_process.quickstart import Build_Dev_DataLoader
from transformers import AutoModel
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
import numpy as np

def evaluate(net: torch.nn.Module, eval_mind_dataset, device: torch.device):
    net.eval()
    EVAL_BATCH_SIZE = 1
    eval_dataloader = DataLoader(eval_mind_dataset, batch_size=EVAL_BATCH_SIZE, pin_memory=True)

    val_metrics_list = []
    for batch in tqdm(eval_dataloader, desc="Evaluation for MINDValDataset"):
        # Inference
        batch["histories_text"] = batch["histories_text"].to(device)
        batch["histories_mask"] = batch["histories_mask"].to(device)
        batch["histories_imgs"] = batch["histories_imgs"].to(device)
        batch["condidate_text"] = batch["condidate_text"].to(device)
        batch["condidate_mask"] = batch["condidate_mask"].to(device)
        batch["condidate_imgs"] = batch["condidate_imgs"].to(device)
        batch["labels"] = batch["labels"].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["labels"].flatten().cpu().to(torch.int).numpy()

        # Calculate Metrics
        val_metrics_list.append(RecEvaluator.evaluate_all(y_true, y_score))

    rec_metrics = RecMetrics(
        **{
            "ndcg_at_10": np.average([metrics_item.ndcg_at_10 for metrics_item in val_metrics_list]),
            "ndcg_at_5": np.average([metrics_item.ndcg_at_5 for metrics_item in val_metrics_list]),
            "auc": np.average([metrics_item.auc for metrics_item in val_metrics_list]),
            "mrr": np.average([metrics_item.mrr for metrics_item in val_metrics_list]),
        }
    )

    return rec_metrics

device = "cuda:0"

nrms_net = get_NRMS().to(device, dtype = torch.bfloat16)
path_to_model = "autodl-tmp/output/model/2024-05-04/15-50-44/checkpoint-4905/model.safetensors"
# nrms_net.load_state_dict(torch.load(path_to_model))
# tensors = {}
# with safe_open(path_to_model, framework="pt", device=0) as f:
#     for k in f.keys():
#         tensors[k] = f.get_tensor(k)
load_model(nrms_net,path_to_model)

dev_dataset, _ = Build_Dev_DataLoader()

rec_metrics = evaluate(nrms_net, dev_dataset, device = "cuda:0")
print(rec_metrics)