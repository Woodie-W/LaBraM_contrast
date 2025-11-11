# --------------------------------------------------------
# Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI
# By Wei-Bang Jiang
# Based on BEiT-v2, timm, DeiT, DINO, and BIOT code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# https://github.com/ycq091044/BIOT
# ---------------------------------------------------------

import io
import os
import math
import time
import json
import glob
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
import pickle
from pathlib import Path
import argparse
import fnmatch
import torch
import torch.distributed as dist
from torch import inf
import h5py

from tensorboardX import SummaryWriter
from data_processor.dataset import ShockDataset
from data_processor.dataset import SFTSet
import pickle
from scipy.signal import resample
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

standard_1020 = [
        'Fp1', 'Fpz', 'Fp2',
        'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
        'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', \
        'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
        'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10', \
        'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
        'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', \
        'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
        'O1', 'Oz', 'O2', 'O9', 'CB1', 'CB2', \
        'Iz', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10',
        'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26',
        'E27', 'E28', 'E29', 'E30',
        'E31', 'E32', 'E33', 'E34', 'E35', 'E36', 'E37', 'E38', 'E39', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46',
        'E47', 'E48', 'E49', 'E50',
        'E51', 'E52', 'E53', 'E54', 'E55', 'E56', 'E57', 'E58', 'E59', 'E60', 'E61', 'E62', 'E63', 'E64', 'E65', 'E66',
        'E67', 'E68', 'E69', 'E70',
        'E71', 'E72', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86',
        'E87', 'E88', 'E89', 'E90',
        'E91', 'E92', 'E93', 'E94', 'E95', 'E96', 'E97', 'E98', 'E99', 'E100', 'E101', 'E102', 'E103', 'E104', 'E105',
        'E106', 'E107', 'E108',
        'E109', 'E110', 'E111', 'E112', 'E113', 'E114', 'E115', 'E116', 'E117', 'E118', 'E119', 'E120', 'E121', 'E122',
        'E123', 'E124', 'E125',
        'E126', 'E127', 'E128', "Vertex Reference",
        'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
        'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
        'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
                median=self.median,
                avg=self.avg,
                global_avg=self.global_avg,
                max=self.max,
                value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                    "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def update_image(self, head='images', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self.writer.add_image(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False):
    world_size = get_world_size()

    if world_size == 1:
        return tensor
    dist.all_reduce(tensor, op=op, async_op=async_op)

    return tensor


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
                tensor_all,
                tensor,
                async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def _get_rank_env():
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_RANK'])


def _get_local_rank_env():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])


def _get_world_size_env():
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    else:
        return int(os.environ['OMPI_COMM_WORLD_SIZE'])


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = _get_rank_env()
        args.world_size = _get_world_size_env()  # int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = _get_local_rank_env()
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
            args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
        module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 layer_names=None):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters, layer_names=layer_names)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0, layer_names=None) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device

    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        # total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        layer_norm = torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters])
        total_norm = torch.norm(layer_norm, norm_type)
        # print(layer_norm.max(dim=0))

        if layer_names is not None:
            if torch.isnan(total_norm) or torch.isinf(total_norm) or total_norm > 1.0:
                value_top, name_top = torch.topk(layer_norm, k=5)
                print(f"Top norm value: {value_top}")
                print(f"Top norm name: {[layer_names[i][7:] for i in name_top.tolist()]}")

    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
            [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in
             iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None,
               save_ckpt_freq=1):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if not getattr(args, 'enable_deepspeed', False):
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        if epoch == 'best':
            checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name), ]
        elif (epoch + 1) % save_ckpt_freq == 0:
            checkpoint_paths.append(output_dir / ('checkpoint-%s.pth' % epoch_name))

        for checkpoint_path in checkpoint_paths:
            to_save = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    # 'scaler': loss_scaler.state_dict(),
                    'args': args,
            }
            if loss_scaler is not None:
                to_save['scaler'] = loss_scaler.state_dict()

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            if optimizer_disc is not None:
                to_save['optimizer_disc'] = optimizer_disc.state_dict()

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, optimizer_disc=None):
    output_dir = Path(args.output_dir)

    if not getattr(args, 'enable_deepspeed', False):
        # torch.amp
        if args.auto_resume and len(args.resume) == 0:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint.pth'))
            if len(all_checkpoints) > 0:
                args.resume = os.path.join(output_dir, 'checkpoint.pth')
            else:
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

        if args.resume:
            if args.resume.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                        args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])  # strict: bool=True, , strict=False
            print("Resume checkpoint %s" % args.resume)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resume checkpoint at epoch {checkpoint['epoch']}")

                # 这里做了修改
                # args.start_epoch = 1  # checkpoint['epoch'] + 1
                args.start_epoch = checkpoint['epoch'] + 1

                if hasattr(args, 'model_ema') and args.model_ema:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
            if 'optimizer_disc' in checkpoint:
                optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
    else:
        # deepspeed, only support '--auto_resume'.
        if args.auto_resume:
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                args.resume = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(args.output_dir, tag='checkpoint-%d' % latest_ckpt)
                args.start_epoch = client_states['epoch'] + 1
                if model_ema is not None:
                    if args.model_ema:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def create_ds_config(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "latest"), mode="w") as f:
        pass

    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
                "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
                "train_micro_batch_size_per_gpu": args.batch_size,
                "steps_per_print": 1000,
                "optimizer": {
                        "type": "Adam",
                        "adam_w_mode": True,
                        "params": {
                                "lr": args.lr,
                                "weight_decay": args.weight_decay,
                                "bias_correction": True,
                                "betas": [
                                        0.9,
                                        0.999
                                ],
                                "eps": 1e-8
                        }
                },
                "fp16": {
                        "enabled": True,
                        "loss_scale": 0,
                        "initial_scale_power": 7,
                        "loss_scale_window": 128
                }
        }

        writer.write(json.dumps(ds_config, indent=2))


def build_pretraining_dataset(datasets: list, time_window: list, stride_size=200, start_percentage=0, end_percentage=1):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list, window_size in zip(datasets, time_window):
        dataset = ShockDataset([Path(file_path) for file_path in dataset_list], window_size * 200, stride_size,
                               start_percentage, end_percentage)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list


def build_finetuning_dataset(datasets: list,label2int=None,kept_ints=None):
    shock_dataset_list = []
    ch_names_list = []
    for dataset_list in datasets:
        dataset = SFTSet(data_path=None, dataset_path_without_cls_list=dataset_list, clip=True,label2int=label2int,kept_ints=kept_ints)
        shock_dataset_list.append(dataset)
        ch_names_list.append(dataset.get_ch_names())
    return shock_dataset_list, ch_names_list


def get_input_chans(ch_names):
    input_chans = [0]  # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


class TUEVLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y

# 递归找到这个文件夹下的所有pkl文件和channel_name文件
def find_data_files(base_folder):
    data_files = []
    channel_name_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if fnmatch.fnmatch(file, '*.pkl') and 'channel_name' not in file:
                data_files.append(os.path.join(root, file))
            if 'channel_name' in file:
                channel_name_files.append(os.path.join(root, file))
    return data_files,channel_name_files

def split_files(data_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    np.random.shuffle(data_files)
    total_files = len(data_files)
    
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    train_files = data_files[:train_end]
    val_files = data_files[train_end:val_end]
    test_files = data_files[val_end:]
    
    return train_files, val_files, test_files

class TUHEEG_PROCESSED_Loader(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        data = pickle.load(open(self.files[index], "rb"))
        data = torch.FloatTensor(data)

        if "/eval/abnormal/" in self.files[index] or "/train/abnormal/" in self.files[index]:
            label = 1  # 1 represent abnormal
        elif "/eval/normal/" in self.files[index] or "/train/normal/" in self.files[index]:
            label = 0  # 0 represent normal
        else:
            raise ValueError(f"Unexpected label in file path: {self.files[index]}")
        
        return data,label

def prepare_TUHEEG_PROCESSED_dataset(root):
    seed = 12345
    np.random.seed(seed)
    val_test_data_files,channel_name_files = find_data_files(os.path.join(root,'tuh_eeg_abnormal','eval'))
    print(os.path.join(root,'tuh_eeg_abnormal','eval'))
    _,val_files, test_files = split_files(val_test_data_files, train_ratio=0, val_ratio=0.5, test_ratio=0.5)
    ch_names = pickle.load(open(channel_name_files[0], "rb"))
    val_dataset = TUHEEG_PROCESSED_Loader(val_files)
    test_dataset = TUHEEG_PROCESSED_Loader(test_files)

    train_data_files,_ = find_data_files(os.path.join(root,'tuh_eeg_abnormal','train'))
    train_dataset = TUHEEG_PROCESSED_Loader(train_data_files)
    # print("train_data_files: ",len(train_data_files),"val_files: ", len(val_files),"test_files: ", len(test_files))
    return train_dataset,test_dataset, val_dataset,ch_names


# class SFTSet(Dataset):
#     def __init__(self, data_path, data_path_without_cls, clip=False, label2int=None, kept_ints=None):
#         """
#         :param data_path: a dict, each contains data from a cls,
#             e,g.: {'HC': [00001, 00002,], 'MDD': [00100, 00101]}
#         :param data_path_without_cls: a list, the cls of which has to be read from f'{total_id}_info.pkl'
#             e.g. [00200, 00201, ...]
#         """
#         self.data_path = data_path
#         self.data_path_without_cls = data_path_without_cls
#         self.file_paths = []  # paths of each chunk
#         self.class_labels = []  # for subj in data_path, len(class_labels) < len(file_paths)

#         self.clip = clip

#         channel_paths = []  # paths of channel_locs of each subject
#         label_paths = []  # 1 for eeg, 0 for fnirs
#         info_paths = []

#         # 遍历文件夹,获取所有 pkl 文件路径
#         if data_path is not None:
#             for class_name in data_path:
#                 for subj_path in data_path[class_name]:
#                     if os.path.isdir(subj_path):
#                         for file in os.listdir(subj_path):
#                             if file.endswith('.pkl') and 'features' in file:
#                                 self.file_paths.append(os.path.join(subj_path, file))
#                                 self.class_labels.append(class_name)
#                             elif file.endswith('.pkl') and 'info' in file:
#                                 info_paths.append(os.path.join(subj_path, file))
#                             elif file.endswith('.pkl') and 'channel' in file:
#                                 channel_paths.append(os.path.join(subj_path, file))
#                             elif file.endswith('.pkl') and 'label' in file:
#                                 label_paths.append(os.path.join(subj_path, file))

#         # 遍历文件夹,获取所有 pkl 文件路径  for data_path_without_cls
#         if data_path_without_cls is not None:
#             for subj_path in data_path_without_cls:
#                 if os.path.isdir(subj_path):
#                     for file in os.listdir(subj_path):
#                         if file.endswith('.pkl') and 'features' in file:
#                             self.file_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'info' in file:
#                             info_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'channel' in file:
#                             channel_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'label' in file:
#                             label_paths.append(os.path.join(subj_path, file))

#         self.id2label_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in label_paths])
#         self.id2channel_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in channel_paths])
#         self.id2info_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in info_paths])

#         self.sub2label, label2counts = self.analyse_label()
#         print('Distribution of cls labels', label2counts, 'num samples', len(self.file_paths))

#         # todo: 需要检查，后续可能增加
#         if label2int is None:
#             self.label2int = {
#                 'HC': 0, 'HEALTHY': 0, 'LowOCD': 0,
#                 'AD': 1, 'FTD': 2,
#                 'PD': 3, 'PARKINSON': 3,
#                 'past-MDD': 4, 'MDD': 5, 'Dp': 6,
#                 'ADHD': 7, 'ADHD ': 7,
#                 'OCD': 8, 'HighOCD': 8,
#                 'SMC': 9, 'CHRONIC PAIN': 10, 'MSA-C': 11, 'DYSLEXIA': 12, 'TINNITUS': 13,
#                 'INSOMNIA': 14, 'BURNOUT': 15, 'DEPERSONALIZATION': 16, 'ANXIETY': 17, 'BIPOLAR': 18,
#                 'PDD NOS ': 19, 'PDD NOS': 19,
#                 'ASD': 20, 'ASPERGER': 21, 'MIGRAINE': 22, 'PANIC': 23, 'TUMOR': 24,
#                 'WHIPLASH': 25, 'PAIN': 26, 'CONVERSION DX': 27,
#                 'STROKE ': 28, 'STROKE': 28,
#                 'LYME': 29, 'PTSD': 30,
#                 'EPILEPSY': 31, 'abnormal': 31,
#                 'TRAUMA': 32, 'TBI': 33, 'DPS': 34, 'ANOREXIA': 35, 'DYSPRAXIA': 36,
#                 'DYSCALCULIA': 37, 'GTS': 38,
#                 'mTBI': 39,
#                 'SZ': 40,
#                 'A&A': 41,
#                 'Delirium': 42,
#                 'PD-FOG-': 43, 'PD-FOG+': 44,
#                 'Chronic TBI': 45,
#                 'Recrudesce': 46, 'Somatic': 47,  # todo
#             }
#         else:
#             self.label2int = label2int

#         # Delete data with Unknown label
#         for idx in reversed(range(len(self.file_paths))):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             if self.sub2label[int(total_id)] in ['unknown', None, ['unknown']]:
#                 del self.file_paths[idx]
#                 if idx < len(self.class_labels):
#                     del self.class_labels[idx]
#             elif kept_ints is not None:
#                 ints = [self.label2int[l] for l in get_multi_label(self.sub2label[int(total_id)])]
#                 used_flag = False
#                 for i in ints:
#                     if i in used_ints:
#                         used_flag = True
#                 if not used_flag:
#                     del self.file_paths[idx]
#                     if idx < len(self.class_labels):
#                         del self.class_labels[idx]

#         print('num samples, without unknown label', len(self.file_paths))

#         # Viz distribution of labels
#         int2counts = dict()
#         for label in label2counts:  # todo: 检查label2counts， 应该避免重复被试，按照subject_dataset_id计算，如果某个标签对应的被试只有一个就无法对比
#             if label not in self.label2int:
#                 continue
#             if self.label2int[label] in int2counts:
#                 int2counts[self.label2int[label]] += label2counts[label]
#             else:
#                 int2counts[self.label2int[label]] = label2counts[label]
#         int_counts = sorted([(k, v) for k, v in int2counts.items()], key=lambda x: x[1], reverse=True)
#         print('Distribution of cls labels (without unknown):')
#         for i in int_counts:
#             print('label', i[0], [k for k in self.label2int if self.label2int[k] == i[0]], 'counts', i[1],)

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):

#         # 加载对应索引的 pkl 文件
#         with open(self.file_paths[idx], 'rb') as f:
#             de_features = pickle.load(f)
#         total_id = self.file_paths[idx].split('/')[-1].split('_')[0]

#         # 加载对应的通道名称文件,比较被试ID
#         with open(self.id2channel_path[int(total_id)], 'rb') as f:
#             channels = pickle.load(f)

#         # 加载对应的标签文件，比较被试ID
#         with open(self.id2label_path[int(total_id)], 'rb') as f:
#             labels = pickle.load(f)

#         if idx < len(self.class_labels):  # if class labels is provided,
#             class_label = self.class_labels[idx]
#         else:
#             assert int(total_id) in self.id2info_path
#             with open(self.id2info_path[int(total_id)], 'rb') as f:
#                 info = pickle.load(f)
#                 class_label = info['subject_label']
#         if isinstance(class_label, list):
#             class_label = [self.label2int[cl] for cl in class_label]
#         else:
#             assert isinstance(class_label, str)
#             class_label = [self.label2int[class_label]]

#         # # 转换 class_label 为二分类标签
#         # class_label_binary = [0 if cl == 0 else 1 for cl in class_label]

#         if self.clip:
#             # todo: should be the same as the snippet in Subject_data_new
#             mean = de_features.mean(axis=1, keepdims=True)
#             std = de_features.std(axis=1, keepdims=True)
#             de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
#             threshold = 30
#             de_features = np.clip(de_features, -threshold, threshold)
#         length = min(len(de_features), len(channels))
#         return de_features, channels, labels, int(total_id), class_label

#     def analyse_label(self):
#         # 获取疾病标签的分布
#         # collect all possible cls labels
#         sub2label = dict()
#         for idx in range(len(self.class_labels)):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             sub2label[int(total_id)] = self.class_labels[idx]
#         for total_id in self.id2info_path:
#             if total_id in sub2label:
#                 continue
#             with open(self.id2info_path[total_id], 'rb') as f:
#                 sub2label[total_id] = pickle.load(f)['subject_label']

#         label2counts = dict()
#         for sub, label in sub2label.items():
#             new_items = get_multi_label(label)
#             for item in new_items:
#                 if item in label2counts:
#                     label2counts[item] += 1
#                 else:
#                     label2counts[item] = 1

#         return sub2label, label2counts

class SXMU_2_PROCESSED_Set(torch.utils.data.Dataset):
    def __init__(self, data_path, clip=False, label2int=None):
        self.data_path = data_path
        self.file_paths = []  # paths of each chunk
        self.class_labels = []  # class labels for each chunk

        self.clip = clip
        self.label2int = label2int or {}

        # 遍历 data_path 中的每个类别和路径
        for class_name, subjects in data_path.items():
            for subj_path in subjects:
                # print(subj_path)
                if os.path.isdir(subj_path):
                    for file in os.listdir(subj_path):
                        if file.endswith('.pkl') and 'data' in file:
                            self.file_paths.append(os.path.join(subj_path, file))
                            self.class_labels.append(class_name)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        class_label = self.class_labels[idx]

        # Load the data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        if self.clip:
            mean = data.mean(axis=1, keepdims=True)
            std = data.std(axis=1, keepdims=True)
            data = np.clip(data, mean - std * 3, mean + std * 3)
            threshold = 30
            data = np.clip(data, -threshold, threshold)

        # Convert class label to integer if mapping exists
        label_int = self.label2int.get(class_label, -1)

        return data, label_int

def prepare_SXMU_2_PROCESSED_dataset(root,args):
    seed = 12345
    np.random.seed(seed)
    hc_path = os.path.join(root, 'HC', '59chs')
    mdd_path = os.path.join(root, 'MDD', '59chs')
    
    # 将相同身份的人放在一起
    hc_subject_id_dict = {}
    for subj in os.listdir(hc_path):
        if not subj.startswith('channel_name'):
            subj_path = os.path.join(hc_path, subj)
            info_file = os.path.join(subj_path, f"{subj}_info.pkl")
            with open(info_file, 'rb') as f:
                info_dict = pickle.load(f)
            subject_id = info_dict['subject_id_dateset']
            if subject_id not in hc_subject_id_dict:
                hc_subject_id_dict[subject_id] = []
            hc_subject_id_dict[subject_id].append(subj_path)
        
    mdd_subject_id_dict = {}
    for subj in os.listdir(mdd_path):
        if not subj.startswith('channel_name'):
            subj_path = os.path.join(mdd_path, subj)
            info_file = os.path.join(subj_path, f"{subj}_info.pkl")
            with open(info_file, 'rb') as f:
                info_dict = pickle.load(f)
            subject_id = info_dict['subject_id_dateset']
            if subject_id not in mdd_subject_id_dict:
                mdd_subject_id_dict[subject_id] = []
            mdd_subject_id_dict[subject_id].append(subj_path)
        
    # 输出被试数和文件夹数
    hc_num_subjects = len(hc_subject_id_dict)
    hc_num_folders = sum(len(folders) for folders in hc_subject_id_dict.values())
    # print(f"Number of hc subjects: {num_subjects}")
    # print(f"Number of hc folders: {num_folders}")

    # 输出被试数和文件夹数
    mdd_num_subjects = len(mdd_subject_id_dict)
    mdd_num_folders = sum(len(folders) for folders in mdd_subject_id_dict.values())
    # print(f"Number of mdd subjects: {num_subjects}")
    # print(f"Number of mdd folders: {num_folders}")

    hc_subject_ids = list(hc_subject_id_dict.keys())
    mdd_subject_ids = list(mdd_subject_id_dict.keys())
    hc_kfold_indices = np.array_split(np.random.permutation(hc_subject_ids), 10)
    mdd_kfold_indices = np.array_split(np.random.permutation(mdd_subject_ids), 10)
    
    print(f"Fold {args.fold + 1}/10")

    train_hc_indices = []
    train_mdd_indices = []
    eval_indices = {'HC': [], 'MDD': []}
    test_indices = {'HC': [], 'MDD': []}
    
    for i in range(10):
        if i == args.fold:
            eval_hc_subject_ids = hc_kfold_indices[i].tolist()
            eval_mdd_subject_ids = mdd_kfold_indices[i].tolist()
            
            half_hc = len(eval_hc_subject_ids) // 2
            half_mdd = len(eval_mdd_subject_ids) // 2

            eval_indices['HC'].extend(path for id in eval_hc_subject_ids[:half_hc] for path in hc_subject_id_dict[id])
            eval_indices['MDD'].extend(path for id in eval_mdd_subject_ids[:half_mdd] for path in mdd_subject_id_dict[id])

            test_indices['HC'].extend(path for id in eval_hc_subject_ids[half_hc:] for path in hc_subject_id_dict[id])
            test_indices['MDD'].extend(path for id in eval_mdd_subject_ids[half_mdd:] for path in mdd_subject_id_dict[id])
        else:
            train_hc_subject_ids = hc_kfold_indices[i].tolist()
            train_mdd_subject_ids = mdd_kfold_indices[i].tolist()
            for subject_id in train_hc_subject_ids:
                train_hc_indices.extend(hc_subject_id_dict[subject_id])
            for subject_id in train_mdd_subject_ids:
                train_mdd_indices.extend(mdd_subject_id_dict[subject_id])

    train_indices = {'HC': train_hc_indices, 'MDD': train_mdd_indices}

    # clip = False 暂时不进行裁剪
    train_data = SXMU_2_PROCESSED_Set(data_path=train_indices, clip=False, label2int={'HC': 0, 'MDD': 1})
    test_data = SXMU_2_PROCESSED_Set(data_path=test_indices, clip=False, label2int={'HC': 0, 'MDD': 1})
    eval_data = SXMU_2_PROCESSED_Set(data_path=eval_indices, clip=False, label2int={'HC': 0, 'MDD': 1})

    print("train_data:", train_data.__len__())
    print("test_data:", eval_data.__len__())
    
    ch_names = pickle.load(open(os.path.join(hc_path, 'channel_name.pkl'), "rb"))
    
    return train_data,test_data, eval_data,ch_names 

# class SFTSet(torch.utils.data.Dataset):
#     def __init__(self, data_path, data_path_without_cls, clip=False, label2int=None):
#         """
#         :param data_path: a dict, each contains data from a cls,
#             e.g.: {'HC': [00001, 00002,], 'MDD': [00100, 00101]}
#         :param data_path_without_cls: a list, the cls of which has to be read from f'{total_id}_info.pkl'
#             e.g. [00200, 00201, ...]
#         """
#         self.data_path = data_path
#         self.data_path_without_cls = data_path_without_cls
#         self.file_paths = []  # paths of each chunk
#         self.class_labels = []  # for subj in data_path, len(class_labels) < len(file_paths)

#         self.clip = clip
#         self.label2int = label2int or {}

#         # Traverse directories to gather all .pkl file paths
#         if data_path:
#             for class_name, subjects in data_path.items():
#                 for subj_path in subjects:
#                     if os.path.isdir(subj_path):
#                         for file in os.listdir(subj_path):
#                             if file.endswith('.pkl') and 'data' in file:
#                                 self.file_paths.append(os.path.join(subj_path, file))
#                                 self.class_labels.append(class_name)

#         if data_path_without_cls:
#             for subj_path in data_path_without_cls:
#                 if os.path.isdir(subj_path):
#                     for file in os.listdir(subj_path):
#                         if file.endswith('.pkl') and 'data' in file:
#                             self.file_paths.append(os.path.join(subj_path, file))
#                             # Load class labels from corresponding info files
#                             info_path = os.path.join(subj_path, f'{subj_path.split("/")[-1]}_info.pkl')
#                             with open(info_path, 'rb') as f:
#                                 info = pickle.load(f)
#                                 self.class_labels.append(info['subject_label'])

#         print('Number of samples:', len(self.file_paths))

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):
#         # Load the feature data from the file
#         with open(self.file_paths[idx], 'rb') as f:
#             features = pickle.load(f)

#         # Clip features if specified
#         if self.clip:
#             mean = features.mean(axis=1, keepdims=True)
#             std = features.std(axis=1, keepdims=True)
#             features = np.clip(features, mean - std * 3, mean + std * 3)
#             threshold = 30
#             features = np.clip(features, -threshold, threshold)

#         # Obtain class label and convert to integer using label2int dictionary
#         class_label = self.class_labels[idx]
#         if isinstance(class_label, list):
#             class_label = [self.label2int[cl] for cl in class_label if cl in self.label2int]
#         elif class_label in self.label2int:
#             class_label = self.label2int[class_label]

#         return features, class_label

def prepare_AD_FD_HC_PROCESSED_dataset(root,args):
    seed = 12345
    np.random.seed(seed)
    data_path = os.path.join(root, '19chs')
    label2int = {'HC': 0, 'AD': 1, 'FTD': 2}
    
    subject_id_dict = {}
    for subj in os.listdir(data_path):
        if not subj.startswith('channel_name'):
            subj_path = os.path.join(data_path, subj)
            info_file = os.path.join(subj_path, f"{subj}_info.pkl")
            with open(info_file, 'rb') as f:
                info_dict = pickle.load(f)
            subject_id = info_dict['subject_id_dateset']
            if subject_id not in subject_id_dict:
                subject_id_dict[subject_id] = []
            subject_id_dict[subject_id].append(subj_path)
    
    num_subjects = len(subject_id_dict)
    num_folders = sum(len(folders) for folders in subject_id_dict.values())
    print(f"Number of subjects: {num_subjects}")
    print(f"Number of folders: {num_folders}")
    
    subject_ids = list(subject_id_dict.keys())
    kfold_indices = np.array_split(np.random.permutation(subject_ids), 10)
    
    print(f"Fold {args.fold + 1}/10")
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(10):
        if i == args.fold:
            eval_subject_ids = kfold_indices[i].tolist()
            half = len(eval_subject_ids) // 2
            val_indices.extend(subject_id_dict[sid] for sid in eval_subject_ids[:half])
            test_indices.extend(subject_id_dict[sid] for sid in eval_subject_ids[half:])
        else:
            train_subject_ids = kfold_indices[i].tolist()
            train_indices.extend(subject_id_dict[sid] for sid in train_subject_ids)
    val_indices = [item for sublist in val_indices for item in sublist]
    test_indices = [item for sublist in test_indices for item in sublist]
    train_indices = [item for sublist in train_indices for item in sublist]
    
    train_data = SFTSet(data_path=None, data_path_without_cls=train_indices, clip=False, label2int=label2int)
    val_data = SFTSet(data_path=None, data_path_without_cls=val_indices, clip=False, label2int=label2int)
    test_data = SFTSet(data_path=None, data_path_without_cls=test_indices, clip=False, label2int=label2int)
    
    ch_names = pickle.load(open(os.path.join(data_path, 'channel_name.pkl'), "rb"))
    
    return train_data,test_data, val_data,ch_names

def prepare_resting_AD_FD_HC_PROCESSED_dataset(root,args):
    seed = 12345
    np.random.seed(seed)
    data_path = os.path.join(root, '19chs')
    label2int = {'HC': 0, 'AD': 1, 'FTD': 2}
    
    subjects = [os.path.join(data_path, subj) for subj in os.listdir(data_path)]
    kfold_indices = np.array_split(np.random.permutation(subjects), 10)
    
    print(f"Fold {args.fold + 1}/10")
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(10):
        if i == args.fold:
            eval_indices = kfold_indices[i].tolist()
            half_point = len(eval_indices) // 2
            val_indices = eval_indices[:half_point]
            test_indices = eval_indices[half_point:]
        else:
            train_indices.extend(kfold_indices[i].tolist())
    
    train_data = SFTSet(data_path=None, data_path_without_cls=train_indices, clip=False, label2int=label2int)
    val_data = SFTSet(data_path=None, data_path_without_cls=val_indices, clip=False, label2int=label2int)
    test_data = SFTSet(data_path=None, data_path_without_cls=test_indices, clip=False, label2int=label2int)
    
    ch_names = pickle.load(open(os.path.join(data_path, 'channel_name.pkl'), "rb"))

    return train_data,test_data, val_data, ch_names
    
def prepare_resting_PREDICT_Depression_Rest_dataset(root,args):
    seed = 12345
    np.random.seed(seed)
    data_path = os.path.join(root, '60chs')
    label2int = {'MDD': 0, 'past-MDD': 1, 'Dp': 2, 'HC': 3}
    
    subjects = [os.path.join(data_path, subj) for subj in os.listdir(data_path)]
    kfold_indices = np.array_split(np.random.permutation(subjects), 10)
    
    print(f"Fold {args.fold + 1}/10")
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(10):
        if i == args.fold:
            eval_indices = kfold_indices[i].tolist()
            half_point = len(eval_indices) // 2
            val_indices = eval_indices[:half_point]
            test_indices = eval_indices[half_point:]
        else:
            train_indices.extend(kfold_indices[i].tolist())
            
    train_data = SFTSet(data_path=None, data_path_without_cls=train_indices, clip=False, label2int=label2int)
    val_data = SFTSet(data_path=None, data_path_without_cls=val_indices, clip=False, label2int=label2int)
    test_data = SFTSet(data_path=None, data_path_without_cls=test_indices, clip=False, label2int=label2int)
    
    ch_names = pickle.load(open(os.path.join(data_path, 'channel_name.pkl'), "rb"))
    
    return train_data, test_data, val_data, ch_names

def prepare_resting_Parkinson_eyes_open_PROCESSED_dataset(root,args):
    seed=12345
    np.random.seed(seed)
    data_path = os.path.join(root, '61chs')
    label2int = {'HC': 0, 'PD': 1}
    
    subjects = [os.path.join(data_path, subj) for subj in os.listdir(data_path)]
    kfold_indices = np.array_split(np.random.permutation(subjects), 10)
    
    print(f"Fold {args.fold + 1}/10")
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(10):
        if i == args.fold:
            eval_indices = kfold_indices[i].tolist()
            half_point = len(eval_indices) // 2
            val_indices = eval_indices[:half_point]
            test_indices = eval_indices[half_point:]
        else:
            train_indices.extend(kfold_indices[i].tolist())
            
    train_data = SFTSet(data_path=None, data_path_without_cls=train_indices, clip=False, label2int=label2int)
    val_data = SFTSet(data_path=None, data_path_without_cls=val_indices, clip=False, label2int=label2int)
    test_data = SFTSet(data_path=None, data_path_without_cls=test_indices, clip=False, label2int=label2int)
    
    ch_names = pickle.load(open(os.path.join(data_path, 'channel_name.pkl'), "rb"))
    
    return train_data, test_data, val_data, ch_names

def prepare_resting_PREDICT_PD_LPC_Rest_dataset(root,args):
    seed=12345
    np.random.seed(seed)
    data_path = os.path.join(root, '61chs')
    label2int = {'HC': 0, 'PD': 1}
    
    subjects = [os.path.join(data_path, subj) for subj in os.listdir(data_path)]
    kfold_indices = np.array_split(np.random.permutation(subjects), 10)
    
    print(f"Fold {args.fold + 1}/10")
    
    train_indices = []
    val_indices = []
    test_indices = []
    for i in range(10):
        if i == args.fold:
            eval_indices = kfold_indices[i].tolist()
            half_point = len(eval_indices) // 2
            val_indices = eval_indices[:half_point]
            test_indices = eval_indices[half_point:]
        else:
            train_indices.extend(kfold_indices[i].tolist())
            
    train_data = SFTSet(data_path=None, data_path_without_cls=train_indices, clip=False, label2int=label2int)
    val_data = SFTSet(data_path=None, data_path_without_cls=val_indices, clip=False, label2int=label2int)
    test_data = SFTSet(data_path=None, data_path_without_cls=test_indices, clip=False, label2int=label2int)
    
    ch_names = pickle.load(open(os.path.join(data_path, 'channel_name.pkl'), "rb"))
    
    return train_data, test_data, val_data, ch_names
    
def prepare_resting_SXMU_2_PROCESSED_dataset(root,args):
    seed = 12345
    np.random.seed(seed)
    hc_path = os.path.join(root, 'HC', '59chs')
    mdd_path = os.path.join(root, 'MDD', '59chs')
    
    hc_subjects = [os.path.join(hc_path, subj) for subj in os.listdir(hc_path)]
    mdd_subjects = [os.path.join(mdd_path, subj) for subj in os.listdir(mdd_path)]
    hc_kfold_indices = np.array_split(np.random.permutation(hc_subjects), 10)
    mdd_kfold_indices = np.array_split(np.random.permutation(mdd_subjects), 10)
    
    print(f"Fold {args.fold + 1}/10")
    
    train_hc_indices = []
    train_mdd_indices = []
    eval_indices = {'HC': [], 'MDD': []}
    test_indices = {'HC': [], 'MDD': []}
    for i in range(10):
        if i == args.fold:
            eval_hc_subjects = hc_kfold_indices[i].tolist()
            eval_mdd_subjects = mdd_kfold_indices[i].tolist()
            half_hc = len(eval_hc_subjects) // 2
            half_mdd = len(eval_mdd_subjects) // 2
            eval_indices['HC'].extend(eval_hc_subjects[:half_hc])
            eval_indices['MDD'].extend(eval_mdd_subjects[:half_mdd])
            test_indices['HC'].extend(eval_hc_subjects[half_hc:])
            test_indices['MDD'].extend(eval_mdd_subjects[half_mdd:])
        else:
            train_hc_subjects = hc_kfold_indices[i].tolist()
            train_mdd_subjects = mdd_kfold_indices[i].tolist()
            train_hc_indices.extend(train_hc_subjects)
            train_mdd_indices.extend(train_mdd_subjects)
            
    train_indices = {'HC': train_hc_indices, 'MDD': train_mdd_indices}
    train_data = SFTSet(data_path=train_indices, data_path_without_cls=None, clip=False, label2int={'HC': 0, 'MDD': 1})
    test_data = SFTSet(data_path=test_indices, data_path_without_cls=None, clip=False, label2int={'HC': 0, 'MDD': 1})
    eval_data = SFTSet(data_path=eval_indices, data_path_without_cls=None, clip=False, label2int={'HC': 0, 'MDD': 1})
    
    ch_names = pickle.load(open(os.path.join(hc_path, 'channel_name.pkl'), "rb"))
    return train_data, test_data, eval_data, ch_names

def prepare_parkinson_dataset(args):
    sft_paths = [
        "./resting_fine_pool/resting_eye_open/Td_eyeopen",
        "./resting_fine_pool/resting_eye_close/Td_eyeclose",
        "./resting_fine_pool/resting_eye_open/Parkinson_eyes_open_PROCESSED",
        "./resting_fine_pool/resting_unknown/PREDICT-PD_LPC_Rest",
        "./resting_fine_pool/resting_eye_open/PREDICT-PD_LPC_Rest_2",
        "./resting_fine_pool/resting_unknown/Parkinson",
    ]
    train_datasets = list(set(sft_paths) - set(val_dataset))
    
    train_data_path_wo_cls = []
    for dataset in train_datasets:
        train_data_path_wo_cls.extend([os.path.join(dataset, file) for file in os.listdir(dataset)])
    
    val_data_path_wo_cls = []
    val_data_path_wo_cls.extend([os.path.join(val_dataset, file) for file in os.listdir(val_dataset)])
    
    np.random.seed(12345)
    
    train_data = SFTSet(data_path=None, data_path_without_cls=train_data_path_wo_cls, clip=False, used_ints=[0,3])
    eval_data = SFTSet(data_path=None, data_path_without_cls=val_data_path_wo_cls, clip=False, used_ints=[0,3])
            

def get_multi_label(label):
    items = []
    if label is None:
        items = [None]
    elif isinstance(label, list):
        for item in label:
            items.append(item)
    else:
        assert isinstance(label, str)
        items = [label]
    return items


# class SFTSet(Dataset):
#     def __init__(self, data_path, data_path_without_cls, clip=False, label2int=None, kept_ints=None):
#         """
#         :param data_path: a dict, each contains data from a cls,
#             e,g.: {'HC': [00001, 00002,], 'MDD': [00100, 00101]}
#         :param data_path_without_cls: a list, the cls of which has to be read from f'{total_id}_info.pkl'
#             e.g. [00200, 00201, ...]
#         """
#         self.data_path = data_path
#         self.data_path_without_cls = data_path_without_cls
#         self.file_paths = []  # paths of each chunk
#         self.class_labels = []  # for subj in data_path, len(class_labels) < len(file_paths)

#         self.clip = clip

#         channel_paths = []  # paths of channel_locs of each subject
#         label_paths = []  # 1 for eeg, 0 for fnirs
#         info_paths = []

#         # 这里file_paths包括了所有数据，class_labels只包括有标签的数据，channel_paths、label_paths、info_paths每个被试占一个元素
#         # 遍历文件夹,获取所有 pkl 文件路径
#         if data_path is not None:
#             for class_name in data_path:
#                 for subj_path in data_path[class_name]:
#                     if os.path.isdir(subj_path):
#                         for file in os.listdir(subj_path):
#                             if file.endswith('.pkl') and 'data' in file:
#                                 self.file_paths.append(os.path.join(subj_path, file))
#                                 self.class_labels.append(class_name)
#                             elif file.endswith('.pkl') and 'info' in file:
#                                 info_paths.append(os.path.join(subj_path, file))
#                             # elif file.endswith('.pkl') and 'channel' in file:
#                             #     channel_paths.append(os.path.join(subj_path, file))
#                             # elif file.endswith('.pkl') and 'label' in file:
#                             #     label_paths.append(os.path.join(subj_path, file))

#         # 遍历文件夹,获取所有 pkl 文件路径  for data_path_without_cls
#         if data_path_without_cls is not None:
#             for subj_path in data_path_without_cls:
#                 if os.path.isdir(subj_path):
#                     for file in os.listdir(subj_path):
#                         if file.endswith('.pkl') and 'data' in file:
#                             self.file_paths.append(os.path.join(subj_path, file))
#                         elif file.endswith('.pkl') and 'info' in file:
#                             info_paths.append(os.path.join(subj_path, file))
#                         # elif file.endswith('.pkl') and 'channel' in file:
#                         #     channel_paths.append(os.path.join(subj_path, file))
#                         # elif file.endswith('.pkl') and 'label' in file:
#                         #     label_paths.append(os.path.join(subj_path, file))

        
#         # 存放被试对应的label、channel、info文件路径
#         # self.id2label_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in label_paths])
#         self.id2channel_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in channel_paths])
#         self.id2info_path = dict([[int(path.split('/')[-1].split('_')[0]), path] for path in info_paths])
        
#         # sub2label被试和对应的标签（单标签和多标签），label2counts标签和对应的数量
#         self.sub2label, label2counts = self.analyse_label()
#         print('Distribution of cls labels', label2counts, 'num samples', len(self.file_paths))

#         # todo: 需要检查，后续可能增加
#         if label2int is None:
#             self.label2int = {
#                 'HC': 0, 'HEALTHY': 0, 'LowOCD': 0,
#                 'AD': 1, 'FTD': 2,
#                 'PD': 3, 'PARKINSON': 3,
#                 'past-MDD': 4, 'MDD': 5, 'Dp': 6,
#                 'ADHD': 7, 'ADHD ': 7,
#                 'OCD': 8, 'HighOCD': 8,
#                 'SMC': 9, 'CHRONIC PAIN': 10, 'MSA-C': 11, 'DYSLEXIA': 12, 'TINNITUS': 13,
#                 'INSOMNIA': 14, 'BURNOUT': 15, 'DEPERSONALIZATION': 16, 'ANXIETY': 17, 'BIPOLAR': 18,
#                 'PDD NOS ': 19, 'PDD NOS': 19,
#                 'ASD': 20, 'ASPERGER': 21, 'MIGRAINE': 22, 'PANIC': 23, 'TUMOR': 24,
#                 'WHIPLASH': 25, 'PAIN': 26, 'CONVERSION DX': 27,
#                 'STROKE ': 28, 'STROKE': 28,
#                 'LYME': 29, 'PTSD': 30,
#                 'EPILEPSY': 31, 'abnormal': 31,
#                 'TRAUMA': 32, 'TBI': 33, 'DPS': 34, 'ANOREXIA': 35, 'DYSPRAXIA': 36,
#                 'DYSCALCULIA': 37, 'GTS': 38,
#                 'mTBI': 39,
#                 'SZ': 40,
#                 'A&A': 41,
#                 'Delirium': 42,
#                 'PD-FOG-': 43, 'PD-FOG+': 44,
#                 'Chronic TBI': 45,
#                 'Recrudesce': 46, 'Somatic': 47,  # todo
#             }
#         else:
#             self.label2int = label2int

#         # Delete data with Unknown label 并且删除不需要的标签
#         for idx in reversed(range(len(self.file_paths))):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             if self.sub2label[int(total_id)] in ['unknown', None, ['unknown']]:
#                 del self.file_paths[idx]
#                 if idx < len(self.class_labels):
#                     del self.class_labels[idx]
#             elif kept_ints is not None:
#                 ints = [self.label2int[l] for l in get_multi_label(self.sub2label[int(total_id)])]
#                 used_flag = False
#                 for i in ints:
#                     if i in kept_ints:
#                         used_flag = True
#                 if not used_flag:
#                     del self.file_paths[idx]
#                     if idx < len(self.class_labels):
#                         del self.class_labels[idx]

#         print('num samples, without unknown label', len(self.file_paths))

#         # Viz distribution of labels
#         int2counts = dict()
#         for label in label2counts:  # todo: 检查label2counts， 应该避免重复被试，按照subject_dataset_id计算，如果某个标签对应的被试只有一个就无法对比
#             if label not in self.label2int:
#                 continue
#             if self.label2int[label] in int2counts:
#                 int2counts[self.label2int[label]] += label2counts[label]
#             else:
#                 int2counts[self.label2int[label]] = label2counts[label]
#         int_counts = sorted([(k, v) for k, v in int2counts.items()], key=lambda x: x[1], reverse=True)
#         print('Distribution of cls labels (without unknown):')
#         for i in int_counts:
#             print('label', i[0], [k for k in self.label2int if self.label2int[k] == i[0]], 'counts', i[1],)
        
#         self.int2counts = int2counts
    
#     def analyse_label(self):
#         # 获取疾病标签的分布
#         # collect all possible cls labels
#         # sub2label: {total_id: class_label} class_label can be a str list(mutli-label) or a string(single label)
#         sub2label = dict()
#         for idx in range(len(self.class_labels)):
#             total_id = self.file_paths[idx].split('/')[-1].split('_')[0]
#             sub2label[int(total_id)] = self.class_labels[idx]
#         for total_id in self.id2info_path:
#             if total_id in sub2label:
#                 continue
#             with open(self.id2info_path[total_id], 'rb') as f:
#                 sub2label[total_id] = pickle.load(f)['subject_label']
#         # label2counts: {class_label(single): counts}
#         label2counts = dict()
#         for sub, label in sub2label.items():
#             # new_items: a list of labels
#             new_items = get_multi_label(label)
#             for item in new_items:
#                 if item in label2counts:
#                     label2counts[item] += 1
#                 else:
#                     label2counts[item] = 1

#         return sub2label, label2counts

#     def __len__(self):
#         return len(self.file_paths)

#     def __getitem__(self, idx):

#         # 加载对应索引的 pkl 文件   total id != idx?
#         with open(self.file_paths[idx], 'rb') as f:
#             de_features = pickle.load(f)
#         total_id = self.file_paths[idx].split('/')[-1].split('_')[0]

#         # 加载对应的通道名称文件,比较被试ID
#         with open(self.id2channel_path[int(total_id)], 'rb') as f:
#             channels = pickle.load(f)

#         # 加载对应的标签文件，比较被试ID
#         with open(self.id2label_path[int(total_id)], 'rb') as f:
#             labels = pickle.load(f)

#         if idx < len(self.class_labels):  # if class labels is provided,
#             class_label = self.class_labels[idx]
#         else:
#             assert int(total_id) in self.id2info_path
#             with open(self.id2info_path[int(total_id)], 'rb') as f:
#                 info = pickle.load(f)
#                 class_label = info['subject_label']

#         # 给定的是list,Multi-label or single label
#         if isinstance(class_label, list):
#             class_label = [self.label2int[cl] for cl in class_label]
#         else:
#             assert isinstance(class_label, str)
#             class_label = [self.label2int[class_label]]

#         if self.clip:
#             # todo: should be the same as the snippet in Subject_data_new
#             mean = de_features.mean(axis=1, keepdims=True)
#             std = de_features.std(axis=1, keepdims=True)
#             de_features = np.clip(de_features, mean - std * 3, mean + std * 3)
#             threshold = 30
#             de_features = np.clip(de_features, -threshold, threshold)
#         length = min(len(de_features), len(channels))
#         # 返回的内容包括：Data(数据):[n_chans,times,feature_dim]，channels(通道位置:三维):[n_chans,3]，labels(1为脑电数据，0为fnirs数据):[n_chans]，total_id(被试ID):int，class_label(标签):int
#         return de_features, channels, labels, int(total_id), class_label
    
    

    
    
def prepare_TUEV_dataset(root):
    # set random seed
    seed = 4523
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "processed_train"))
    val_files = os.listdir(os.path.join(root, "processed_eval"))
    test_files = os.listdir(os.path.join(root, "processed_test"))

    # prepare training and test.py data loader
    train_dataset = TUEVLoader(
            os.path.join(
                    root, "processed_train"), train_files
    )
    test_dataset = TUEVLoader(
            os.path.join(
                    root, "processed_test"), test_files
    )
    val_dataset = TUEVLoader(
            os.path.join(
                    root, "processed_eval"), val_files
    )
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def prepare_TUAB_dataset(root):
    # set random seed
    seed = 12345
    np.random.seed(seed)

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test.py"))

    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test.py data loader
    train_dataset = TUABLoader(os.path.join(root, "train"), train_files)
    test_dataset = TUABLoader(os.path.join(root, "test.py"), test_files)
    val_dataset = TUABLoader(os.path.join(root, "val"), val_files)
    print(len(train_files), len(val_files), len(test_files))
    return train_dataset, test_dataset, val_dataset

def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (
                len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                    target,
                    output,
                    metrics=metrics,
                    threshold=threshold,
            )
        else:
            results = {
                    "accuracy": 0.0,
                    "balanced_accuracy": 0.0,
                    "pr_auc": 0.0,
                    "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
                target, output, metrics=metrics
        )
    return results

if __name__ == '__main__':
    # train_dataset, val_dataset, test_dataset, ch_names = prepare_TUHEEG_PROCESSED_dataset('/data1/wangkuiyu/model_update_code/fine_pool/EEG/TUHEEG_PROCESSED')
    # print(ch_names)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))
    # for i in range(len(train_dataset)):
    #     print(train_dataset.files[i])
    #     print(train_dataset[i][0].shape, train_dataset[i][1])
    args = argparse.Namespace(fold=0)
    # train_dataset, test_dataset, eval_dataset, ch_names = prepare_SXMU_2_PROCESSED_dataset(root = '/data1/wangkuiyu/model_update_code/fine_pool/EEG/SXMU_2_PROCESSED',args = args)
    # print(ch_names)
    # print(len(train_dataset), len(test_dataset), len(eval_dataset))
    # class1 = 0
    # class2 = 0
    # for i in range(len(train_dataset)):
    #     print(train_dataset.file_paths[i])
    #     print(train_dataset[i][0].shape, train_dataset[i][1])
    #     if train_dataset[i][1] == 0:
    #         class1 += 1
    #     elif train_dataset[i][1] == 1:
    #         class2 += 1
    # print(class1,class2)
    # print(class1/(class1+class2))
    # print(class2/(class1+class2))
    
    # train_dataset, test_dataset, eval_dataset,ch_names = prepare_AD_FD_HC_PROCESSED_dataset(root='/data1/wangkuiyu/model_update_code/fine_pool/SFT/AD_FD_HC_PROCESSED',args=args)
    # class1 = 0
    # class2 = 0
    # class3 = 0
    # for i in range(len(train_dataset)):
    #     print(train_dataset.file_paths[i])
    #     print(train_dataset[i][0].shape, train_dataset[i][1])
    #     if train_dataset[i][1] == 0:
    #         class1 += 1
    #     elif train_dataset[i][1] == 1:
    #         class2 += 1
    #     elif train_dataset[i][1] == 2:
    #         class3 += 1
    # print(class1,class2,class3)
    # print(class1/(class1+class2+class3))
    # print(class2/(class1+class2+class3))
    # print(class3/(class1+class2+class3))
    
    # print(ch_names)
    # print(len(train_dataset), len(test_dataset), len(eval_dataset))
    
    # train_dataset, test_dataset, eval_dataset,ch_names = prepare_resting_AD_FD_HC_PROCESSED_dataset(root='/data1/wangkuiyu/model_update_code/resting_fine_pool/resting_eye_close/AD_FD_HC_PROCESSED',args=args)
    # class1 = 0
    # class2 = 0
    # class3 = 0
    # for i in range(len(train_dataset)):
    #     print(train_dataset.file_paths[i])
    #     print(train_dataset[i][0].shape, train_dataset[i][1])
    #     if train_dataset[i][1] == 0:
    #         class1 += 1
    #     elif train_dataset[i][1] == 1:
    #         class2 += 1
    #     elif train_dataset[i][1] == 2:
    #         class3 += 1
    # print(class1,class2,class3)
    # print(class1/(class1+class2+class3))
    # print(class2/(class1+class2+class3))
    # print(class3/(class1+class2+class3))
    # print(ch_names)
    # print(len(train_dataset), len(test_dataset), len(eval_dataset))
    
    def check_unique_labels(dataset):
        unique_labels = set()
        for i in range(len(dataset)):
            _, label = dataset[i]
            unique_labels.add(label)
        return unique_labels

    # # 调用函数
    # train_unique_labels = check_unique_labels(train_dataset)
    # test_unique_labels = check_unique_labels(test_dataset)
    # eval_unique_labels = check_unique_labels(eval_dataset)

    # # 打印结果
    # print("Unique labels in train dataset:", train_unique_labels)
    # print("Unique labels in test dataset:", test_unique_labels)
    # print("Unique labels in eval dataset:", eval_unique_labels)
    
    train_dataset, test_dataset, eval_dataset,ch_names = prepare_resting_PREDICT_Depression_Rest_dataset(root='/data1/wangkuiyu/model_update_code/resting_fine_pool/resting_unknown/PREDICT-Depression_Rest',args=args)
    class1 = 0
    class2 = 0
    class3 = 0
    class4 = 0
    for i in range(len(train_dataset)):
        print(train_dataset.file_paths[i])
        print(train_dataset[i][0].shape, train_dataset[i][1])
        if train_dataset[i][1] == 0:
            class1 += 1
        elif train_dataset[i][1] == 1:
            class2 += 1
        elif train_dataset[i][1] == 2:
            class3 += 1
        elif train_dataset[i][1] == 3:
            class4 += 1
    print(class1,class2,class3,class4)
    print(class1/(class1+class2+class3+class4))
    print(class2/(class1+class2+class3+class4))
    print(class3/(class1+class2+class3+class4))
    print(class4/(class1+class2+class3+class4))
    print(ch_names)
    print(len(train_dataset), len(test_dataset), len(eval_dataset))
    
    # 调用函数
    train_unique_labels = check_unique_labels(train_dataset)
    test_unique_labels = check_unique_labels(test_dataset)
    eval_unique_labels = check_unique_labels(eval_dataset)

    # 打印结果
    print("Unique labels in train dataset:", train_unique_labels)
    print("Unique labels in test dataset:", test_unique_labels)
    print("Unique labels in eval dataset:", eval_unique_labels)