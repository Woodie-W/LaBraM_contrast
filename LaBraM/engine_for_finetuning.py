# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# ---------------------------------------------------------
import math, torch
import sys
from typing import Iterable, Optional
import os
import numpy as np
from timm.utils import ModelEma
import utils
from einops import rearrange
from sklearn.metrics import balanced_accuracy_score


def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):  # DeepSpeed框架中当前损失缩放（Loss Scale）值的工具函数
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch_for_embedding(model: torch.nn.Module, criterion: torch.nn.Module,
                                  data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                                  device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                                  model_ema: Optional[ModelEma] = None, log_writer=None,
                                  start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                                  num_training_steps_per_epoch=None, update_freq=None, ch_names_list=None,
                                  is_binary=True, used_ints=None, args=None):
    # input_chans = None
    # if ch_names is not None:
    #     input_chans = utils.get_input_chans(ch_names)
    if (used_ints is None): used_ints = [0, 1]
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")  # 日志记录器
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    pred = []
    true = []

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0  # DeepSpeed
    else:
        optimizer.zero_grad()  # PyTorch原生AMP

    data_iter_step = 0  # 全局批次计数器
    for i, (data_loader, ch_names) in enumerate(zip(data_loader_list, ch_names_list)):
        print("data_loader: ", data_loader.dataset.get_dataset_name())
        if len(data_loader) == 0: continue
        input_chans = utils.get_input_chans(ch_names)

        for data_iter_step_in_one_dataset, (samples, targets) in enumerate(
                metric_logger.log_every(data_loader, print_freq, header)):
            # print("sample shape: ", samples.shape)            metric_logger.log_every(...) 作用是每print_freq个batch打印一次日志
            step = data_iter_step // update_freq
            # print("step: ", step)
            if step >= num_training_steps_per_epoch:
                print("Reached the maximum number of training steps for this epoch.")
                continue

            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc  调度表存在
            if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)

            targets = targets.to(device, non_blocking=True)

            # 获取单标签数据
            # single_label_mask = (targets >=0).float().sum(dim=-1) == 1
            # if single_label_mask.sum() == 0:
            #     print(f"No single-label samples in this batch{data_iter_step}, skipping.")
            #     single_label_mask = (targets >=0).float().sum(dim=-1) >=1
            # samples = samples[single_label_mask]
            # targets = targets[single_label_mask][:, 0]

            cls_labels_one = torch.where(torch.isin(targets, torch.tensor(used_ints, device=targets.device)),  # todo
                                         targets, -1)
            cls_labels_one = torch.where((cls_labels_one >= 0).float().sum(dim=-1, keepdims=True) == 1,
                                         cls_labels_one, -1)
            cls_labels_one, _ = torch.sort(cls_labels_one, dim=-1, descending=True)  # 把-1排到后面去
            cls_labels_one = cls_labels_one[:, 0]
            samples = samples[cls_labels_one != -1]
            targets = cls_labels_one[cls_labels_one != -1]
            # print(samples.shape, targets.shape)
            if len(targets) == 0:
                raise ValueError(f'No single-label samples in this batch{data_iter_step}')
            if torch.isnan(samples).any() or torch.isnan(targets).any():
                raise ValueError

            if is_binary:
                targets = targets.float().unsqueeze(-1)

            if loss_scaler is None:
                samples = samples.half()
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)
            else:
                with torch.cuda.amp.autocast():
                    loss, output = train_class_batch(
                        model, samples, targets, criterion, input_chans)

            loss_value = loss.item()
            pred.append(output)
            true.append(targets)

            if not math.isfinite(loss_value):
                # print("Loss is {}, stopping training".format(loss_value))
                # sys.exit(1)
                print("found loss nan, ignore it")

            if loss_scaler is None:
                loss /= update_freq
                model.backward(loss)
                model.step()

                if (data_iter_step + 1) % update_freq == 0:
                    # model.zero_grad()
                    # Deepspeed will call step() & model.zero_grad() automatic
                    if model_ema is not None:
                        model_ema.update(model)
                grad_norm = None
                loss_scale_value = get_loss_scale_for_deepspeed(model)
            else:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                loss /= update_freq
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order,
                                        update_grad=(data_iter_step + 1) % update_freq == 0)
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)
                loss_scale_value = loss_scaler.state_dict()["scale"]

            torch.cuda.synchronize()
            data_iter_step += 1

            if is_binary:
                class_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(),
                            targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
            else:
                class_acc = (output.max(-1)[-1] == targets).float().mean()

            metric_logger.update(loss=loss_value)
            metric_logger.update(class_acc=class_acc)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(class_acc=class_acc, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

    pred_cat = torch.cat(pred, dim=0)
    true_cat = torch.cat(true, dim=0)
    pred_cat, true_cat = pred_cat.cpu().detach().numpy(), true_cat.cpu().detach().numpy()

    # 对于二分类，模型输出的是logits，需要先通过sigmoid转为概率，再设置阈值转为类别
    if is_binary:  # 注意：pred_cat是原始logits，不是概率
        if (true_cat.ndim > 1): true_cat = true_cat.flatten()
        if (pred_cat.ndim > 1): pred_cat = pred_cat.flatten()
        probabilities = 1 / (1 + np.exp(-pred_cat))  # 手动计算Sigmoid
        predicted_labels = (probabilities >= 0.5).astype(int)
    else:
        predicted_labels = pred_cat.argmax(-1)

    balanced_acc = balanced_accuracy_score(true_cat, predicted_labels)
    # balanced_acc = balanced_accuracy_score(true_cat, pred_cat.argmax(-1))

    print('balanced acc', balanced_acc)
    print('each cls', np.unique(true_cat, return_counts=True),
          np.unique(pred_cat.argmax(-1), return_counts=True))

    acc_old = (predicted_labels == true_cat).astype(np.float32).mean().item()
    log_writer.update(balanced_acc_old=balanced_acc, head="loss")  # todo: only for single process
    log_writer.update(acc_old=acc_old, head="loss")
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_for_embedding(data_loader_list, model, device, header='Test:', ch_names_list=None,
                           metrics=['acc'], is_binary=True, used_ints=None, args=None):
    # input_chans = None
    # if ch_names is not None:
    #     input_chans = utils.get_input_chans(ch_names)
    if used_ints is None: used_ints = [0, 1]
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []

    for i, (data_loader, ch_names) in enumerate(zip(data_loader_list, ch_names_list)):
        print("data_loader: ", data_loader.dataset.get_dataset_name())
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, batch in enumerate(metric_logger.log_every(data_loader, 30, header)):
            EEG = batch[0]
            target = batch[-1]
            # single_label_mask = (target >=0).float().sum(dim=-1) == 1
            # if single_label_mask.sum() == 0:
            #     continue
            # EEG = EEG[single_label_mask]
            # target = target[single_label_mask][:, 0]
            cls_labels_one = torch.where(torch.isin(target, torch.tensor(used_ints, device=target.device)),  # todo
                                         target, -1)
            cls_labels_one = torch.where((cls_labels_one >= 0).float().sum(dim=-1, keepdims=True) == 1,
                                         cls_labels_one, -1)
            cls_labels_one, _ = torch.sort(cls_labels_one, dim=-1, descending=True)  # 把-1排到后面去
            cls_labels_one = cls_labels_one[:, 0]
            EEG = EEG[cls_labels_one != -1]
            target = cls_labels_one[cls_labels_one != -1]
            if len(target) == 0:
                continue

            EEG = EEG.float().to(device, non_blocking=True) / 100
            EEG = rearrange(EEG, 'B N (A T) -> B N A T', T=200)
            target = target.to(device, non_blocking=True)
            if is_binary:
                target = target.float().unsqueeze(-1)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(EEG, input_chans=input_chans)
                # print('output111111111111',output.shape)
                loss = criterion(output, target)

            if is_binary:
                output = torch.sigmoid(output).cpu()
                # print('output2222222222',output.shape)
            else:
                output = output.cpu()
                # output = torch.softmax(output[..., :10], dim=-1)  # todo 这里pytorch不支持半浮点报错
                output = torch.softmax(output[..., :args.nb_classes].float(), dim=-1)  # todo
            target = target.cpu()

            # results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
            pred.append(output)
            true.append(target)
            batch_size = EEG.shape[0]
            metric_logger.update(loss=loss.item())
            # for key, value in results.items():
            #     metric_logger.meters[key].update(value, n=batch_size)
            # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    print('each cls', true.unique(return_counts=True), pred.argmax(-1).long().unique(return_counts=True))
    pred, true = pred.numpy(), true.numpy()

    if args.output_dir and getattr(args, 'eval', False):
        # 定义保存文件的路径
        epoch_suffix = f"_epoch{args.epoch}" if hasattr(args, 'epoch') and args.epoch != 'best' else ""

        # 使用带 epoch 后缀的新文件名，避免结果被覆盖
        pred_save_path = os.path.join(args.output_dir, f"predictions_fold{args.fold}{epoch_suffix}.npy")
        true_save_path = os.path.join(args.output_dir, f"true_labels_fold{args.fold}{epoch_suffix}.npy")

        # 保存 numpy 数组
        np.save(pred_save_path, pred)
        np.save(true_save_path, true)
        print(f"预测概率矩阵已保存至: {pred_save_path}")
        print(f"真实标签已保存至: {true_save_path}")

    if is_binary:
        if (true.ndim > 1): true = true.flatten()
        if (pred.ndim > 1): pred = pred.flatten()
        predicted_labels = (pred >= 0.5).astype(int)
    else:  # 对于多分类，取概率最高的类别作为预测结果
        if (true.ndim > 1): true = true.flatten()
        predicted_labels = pred.argmax(-1)

    # 使用正确的预测标签计算平衡准确率
    balanced_acc = balanced_accuracy_score(true, predicted_labels)
    # balanced_acc = balanced_accuracy_score(true, pred.argmax(-1))

    ret = utils.get_metrics(pred, true, metrics, is_binary, 0.5)
    ret['loss'] = metric_logger.loss.global_avg
    ret['balanced_accuracy_old'] = balanced_acc
    ret['accuracy_old'] = (predicted_labels == true).astype(np.float32).mean().item()
    # ret['accuracy_old'] = (pred.argmax(-1) == true).astype(np.float32).mean().item()
    print(balanced_acc, ret)
    return ret