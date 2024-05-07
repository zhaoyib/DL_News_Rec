import hydra
import numpy as np
import torch
from config.config import TrainConfig
from const.path import LOG_OUTPUT_DIR, MIND_SMALL_TRAIN_DATASET_DIR, MIND_SMALL_VAL_DATASET_DIR, MODEL_OUTPUT_DIR
from evaluation.RecEvaluator import RecEvaluator, RecMetrics
from data_process.dataframe import read_behavior_df, read_news_df
from data_process.MindDataset import MindTrainDataset, MindDevDataset
from data_process.quickstart import Build_Dev_DataLoader, Build_Train_DataLoader
from model.Multimodal_NewsEncoder import TextEncoder, VisionEncoder, MultiModalEncoder
from model.UserEncoder import UserEncoder
from model.NRMS import NRMS
from model.quickstart import get_NRMS

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainingArguments, AutoImageProcessor
from transformers.modeling_outputs import ModelOutput
from utils.logger import logger_wrapper
from utils.path import generate_folder_name_with_timestamp
from utils.random_seed import set_random_seed
import traceback
import transformers
from safetensors.torch import load_model


def evaluate(net: torch.nn.Module, eval_mind_dataset, device: torch.device) -> RecMetrics:
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
        batch["label"] = batch["label"].to(device)
        with torch.no_grad():
            model_output: ModelOutput = net(**batch)

        # Convert To Numpy
        y_score: torch.Tensor = model_output.logits.flatten().cpu().to(torch.float64).numpy()
        y_true: torch.Tensor = batch["target"].flatten().cpu().to(torch.int).numpy()

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


def train(
    textual_pretrained: str,
    visual_pretrained: str,
    npratio: int,
    history_size: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    max_len: int,
    logger,
    resume = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    logger.info("Start")
    """
    0. Definite Parameters & Functions
    """
    EVAL_BATCH_SIZE = 1
    textual_hidden_size: int = AutoConfig.from_pretrained(textual_pretrained).hidden_size
    visual_hidden_size: int = AutoConfig.from_pretrained(visual_pretrained).hidden_size
    assert textual_hidden_size == visual_hidden_size,\
    f"Please choose the proper models with same hidden size, now \
    the textual model with {textual_hidden_size} dim, visual model\
    with {visual_hidden_size} dim."
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    model_save_dir = generate_folder_name_with_timestamp(MODEL_OUTPUT_DIR)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    """
    1. Init Model
    """
    logger.info("Initializing Model")
    nrms_net = get_NRMS().to(device, dtype = torch.bfloat16)
    if resume:
        path_to_model = "autodl-tmp/output/model/2024-05-02/17-07-58/checkpoint-9810/model.safetensors"
        load_model(nrms_net,path_to_model)
    # textual_encoder = TextEncoder(textual_pretrained)
    # visual_encoder = VisionEncoder(visual_pretrained)
    # news_encoder = MultiModalEncoder(textual_encoder, visual_encoder)
    # user_encoder = UserEncoder(hidden_size=hidden_size)
    # nrms_net = NRMS(news_encoder=news_encoder, user_encoder=user_encoder, hidden_size=hidden_size, loss_fn=loss_fn).to(
    #     device, dtype=torch.bfloat16
    # )
    # it can use the quickstart to get it.
    
    """
    2. Load Data & Create Dataset
    """
    logger.info("Initialize Dataset")
    dev_dataset, _ = Build_Dev_DataLoader()
    train_dataset, _ = Build_Train_DataLoader()
#     train_news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
#     train_behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
#     train_dataset = MINDTrainDataset(train_behavior_df, train_news_df, transform_fn, npratio, history_size, device)

#     val_news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
#     val_behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
#     eval_dataset = MINDValDataset(val_behavior_df, val_news_df, transform_fn, history_size)

    """
    3. Train
    """
    logger.info("Training Start")
    
    training_args = TrainingArguments(
        output_dir=model_save_dir,
        logging_strategy="steps",
        save_total_limit=5,
        lr_scheduler_type="constant",
        weight_decay=weight_decay,
        optim="adamw_torch",
        evaluation_strategy="no",
        save_strategy="epoch",
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=epochs,
        remove_unused_columns=False,
        logging_dir=LOG_OUTPUT_DIR,
        logging_steps=30,
        report_to="tensorboard",
    )
        
    trainer = Trainer(
        model=nrms_net,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    trainer.train()

    """
    4. Evaluate model by Validation Dataset
    """
    logger.info("Evaluation")
    metrics = evaluate(trainer.model, dev_dataset, device)
    logger.info(metrics.dict())


@hydra.main(version_base=None, config_name="train_config")
def main(cfg: TrainConfig) -> None:
    try:
        logger_save_dir = generate_folder_name_with_timestamp(LOG_OUTPUT_DIR)
        logger_save_dir.mkdir(parents=True, exist_ok=True)
        logger_file_name = "logger.log"
        set_random_seed(cfg.random_seed)
        path_to_model = "autodl-tmp/output/model/2024-05-02/17-07-58/checkpoint-9810/model.safetensors"
        
        logger = logger_wrapper("Train",path = logger_save_dir / logger_file_name)
        train(
            cfg.textual_pretrained,
            cfg.visual_pretrained,
            cfg.npratio,
            cfg.history_size,
            cfg.batch_size,
            cfg.gradient_accumulation_steps,
            cfg.epochs,
            cfg.learning_rate,
            cfg.weight_decay,
            cfg.max_len,
            logger,
            resume = True
        )
    except Exception as e:
        logger.error("Exception occurred: %s", e)
        logger.error("Traceback: %s", traceback.format_exc())


if __name__ == "__main__":
    main()