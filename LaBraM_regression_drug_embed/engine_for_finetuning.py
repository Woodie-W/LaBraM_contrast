import math, torch
import sys
from typing import Iterable, Optional
import os
import numpy as np
import pandas as pd 
from timm.utils import ModelEma
import utils
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from swanlab.integration.pytorch_lightning import SwanLabLogger
from utils import drug_name_to_id
import datetime

from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error
import torch.nn.functional as F

# def train_class_batch(model, samples, drug_ids, target, criterion, ch_names):
#     outputs = model(samples, drug_ids, input_chans=ch_names) ## 增加药物id
#     loss = criterion(outputs, target)
#     return loss, outputs

def train_regression_batch(model, samples, drug_ids, hamd0, target_delta_hamd, criterion, ch_names):
    # 为回归任务优化的批次训练函数
    outputs = model(samples, drug_ids, hamd0, input_chans=ch_names)
    loss = criterion(outputs, target_delta_hamd)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):  # DeepSpeed框架中当前损失缩放（Loss Scale）值的工具函数
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch_for_embedding(model: torch.nn.Module, criterion: torch.nn.Module,
                                  data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                                  device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                                  model_ema: Optional[ModelEma] = None, loggers=None, normalizer=None,
                                  start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                                  num_training_steps_per_epoch=None, update_freq=None, ch_names_list=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    data_iter_step = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0: continue
        input_chans = utils.get_input_chans(ch_names)

        for batch_data in metric_logger.log_every(data_loader, print_freq, header):
            samples, delta_hamd_targets, subject_ids, drug_ids, hamd0 = batch_data
            
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch: continue
            it = start_steps + step

            if lr_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = samples.float().to(device, non_blocking=True) / 100
            # if samples.ndim == 4 and samples.shape[3] == 1: samples = samples.squeeze(-1) 
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            delta_hamd_targets = delta_hamd_targets.to(device, non_blocking=True)
            drug_ids = drug_ids.to(device, non_blocking=True)
            hamd0 = hamd0.to(device, non_blocking=True)
            hamd0_normalized = normalizer.normalize_hamd0(hamd0)

            if loss_scaler is None:
                samples = samples.half()
                loss, output = train_regression_batch(model, samples, drug_ids, hamd0_normalized, delta_hamd_targets, criterion, input_chans)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_regression_batch(model, samples, drug_ids, hamd0_normalized, delta_hamd_targets, criterion, input_chans)

            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            if loss_scaler is None:
                loss /= update_freq
                model.backward(loss)
                model.step()
                if (data_iter_step + 1) % update_freq == 0 and model_ema is not None:
                    model_ema.update(model)
                loss_scale_value = get_loss_scale_for_deepspeed(model)
            else:
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None: model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()
            data_iter_step += 1
            
            batch_mae = F.l1_loss(output, delta_hamd_targets).item()
            metric_logger.update(loss=loss_value, mae=batch_mae)
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if loggers is not None:
                metrics = {"train/loss": loss_value, "train/mae": batch_mae, "opt/lr": optimizer.param_groups[0]["lr"]}
                for log in loggers: log.log_metrics(metrics, step=it)
                    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_for_embedding(data_loader_list, model, device, header='Test:', ch_names_list=None, normalizer=None, args=None, **kwargs):
    criterion = torch.nn.HuberLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()

    all_preds, all_labels, all_subject_ids, all_drug_ids = [], [], [], []

    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0: continue
        input_chans = utils.get_input_chans(ch_names)
        
        for batch_data in metric_logger.log_every(data_loader, 50, header):
            samples, delta_hamd, subject_ids, drug_ids, hamd0 = batch_data
            
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            delta_hamd_gpu = delta_hamd.to(device, non_blocking=True)
            drug_ids_gpu = drug_ids.to(device, non_blocking=True)
            hamd0_gpu = hamd0.to(device, non_blocking=True)

            hamd0_normalized = normalizer.normalize_hamd0(hamd0_gpu)
            
            with torch.cuda.amp.autocast():
                output = model(samples, drug_ids_gpu, hamd0_normalized, input_chans=input_chans)
                loss = criterion(output, delta_hamd_gpu)

            metric_logger.update(loss=loss.item())
            all_preds.append(output.cpu().numpy())
            all_labels.append(delta_hamd.cpu().numpy())
            all_subject_ids.append(subject_ids.cpu().numpy())
            all_drug_ids.append(drug_ids.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_subject_ids = np.concatenate(all_subject_ids)
    all_drug_ids = np.concatenate(all_drug_ids)
    
    results_df = pd.DataFrame({'subject_id': all_subject_ids, 'true_label': all_labels, 'pred_label': all_preds, 'drug_id': all_drug_ids})
    subject_level_results = results_df.groupby('subject_id').agg(true_label=('true_label', 'mean'), pred_label=('pred_label', 'mean'), drug_id=('drug_id', 'first')).reset_index()

    subject_true = subject_level_results['true_label'].values
    subject_pred = subject_level_results['pred_label'].values
    
    mae = float(mean_absolute_error(subject_true, subject_pred))
    r2 = float(r2_score(subject_true, subject_pred))
    pearson_corr, _ = pearsonr(subject_true, subject_pred)
    pearson_corr = float(pearson_corr)

    print("\n--- 被试级数据 ---")
    print(f"* Subject MAE: {mae:.4f}, R2 Score: {r2:.4f}, Pearson Correlation: {pearson_corr:.4f}")
    
    val_metrics_to_log = {"val/MAE": mae, "val/R2": r2, "val/Pearson": pearson_corr, "val/slice_level_loss": metric_logger.loss.global_avg}

    print("\n--- Per-Treatment-Arm Regression Results ---")
    id_to_drug_name = {v: k for k, v in drug_name_to_id.items()}
    for drug_id, group in subject_level_results.groupby('drug_id'):
        drug_name = id_to_drug_name.get(drug_id, f"Unknown_ID_{drug_id}")
        if drug_name == 'unknown' or len(group) < 2: continue
        true, pred = group['true_label'].values, group['pred_label'].values
        drug_mae = float(mean_absolute_error(true, pred))
        drug_r2 = float(r2_score(true, pred))
        drug_pearson, _ = pearsonr(true, pred)
        drug_pearson = float(drug_pearson)
        print(f"  - Arm: {drug_name} ({len(group)} subjects): MAE={drug_mae:.4f}, R2={drug_r2:.4f}, Pearson={drug_pearson:.4f}")
        val_metrics_to_log[f"val_drug/{drug_name}_mae"] = drug_mae
        val_metrics_to_log[f"val_drug/{drug_name}_r2"] = drug_r2
        val_metrics_to_log[f"val_drug/{drug_name}_pearson"] = drug_pearson
        
    print("----------------------------------------------")
    
    ret = {'loss': metric_logger.loss.global_avg, 'mae': mae, 'r2': r2, 'pearson': pearson_corr}
    return ret, val_metrics_to_log, subject_level_results

