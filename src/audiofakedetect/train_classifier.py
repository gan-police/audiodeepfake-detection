"""Source code to train audio deepfake detectors in wavelet space."""

import argparse
import os
import sys
from typing import Any

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.distributed import barrier, destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DiDiP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .data_loader import get_costum_dataset
from .integrated_gradients import (
    Mean,
    integral_approximation,
    interpolate_images,
)
from .models import get_model
from .utils import (
    DotDict,
    _Griderator,
    add_default_parser_args,
    add_noise,
    build_new_grid,
    contrast,
    get_input_dims,
    set_seed,
)
from .wavelet_math import get_transforms


def ddp_setup() -> None:
    """Initialize distributed data parallel environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def create_data_loaders(
    args: DotDict,
    num_workers: int = 8,
) -> tuple:
    """Create the data loaders needed for training.

    Args:
        args (DotDict): Experiment configuration.
        num_workers (int): Number of dataloader workers to use. Defaults to 8.

    Raises:
        NotImplementedError: If args.unknown_prefix is set. This is deprecated.

    Returns:
        tuple: Loaders as a tuple(train loader, validation loader, test loader,
               cross validation loader, cross test loader).
    """
    save_path = args.save_path
    data_path = args.data_path
    limit_train = args.limit_train
    only_use = args.only_use

    train_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="train",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[0],
        asvspoof_name=(
            f"{args.asvspoof_name}_T"
            if args.asvspoof_name is not None and "LA" in args.asvspoof_name
            else args.asvspoof_name
        ),
        file_type=args.file_type,
        resample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    val_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="val",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[1],
        asvspoof_name=(
            f"{args.asvspoof_name}_D"
            if args.asvspoof_name is not None and "LA" in args.asvspoof_name
            else args.asvspoof_name
        ),
        file_type=args.file_type,
        resample_rate=args.sample_rate,
        seconds=args.seconds,
    )
    test_data_set = get_costum_dataset(
        data_path=data_path,
        ds_type="test",
        only_use=only_use,
        save_path=save_path,
        limit=limit_train[2],
        asvspoof_name=(
            f"{args.asvspoof_name}_E"
            if args.asvspoof_name is not None and "LA" in args.asvspoof_name
            else args.asvspoof_name
        ),
        file_type=args.file_type,
        resample_rate=args.sample_rate,
        seconds=args.seconds,
        get_details=args.get_details,
    )
    if args.ddp:
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_data_set, shuffle=True, seed=args.seed, drop_last=True
        )
        val_sampler: DistributedSampler | None = DistributedSampler(
            val_data_set, shuffle=False, seed=args.seed
        )
        test_sampler: DistributedSampler | None = DistributedSampler(
            test_data_set, shuffle=False, seed=args.seed
        )
    else:
        train_sampler = val_sampler = test_sampler = None

    train_data_loader = DataLoader(
        train_data_set,
        batch_size=args.batch_size,
        shuffle=False if args.ddp else True,
        drop_last=True,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_data_loader = DataLoader(
        val_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=num_workers,
        persistent_workers=True,
    )

    test_data_loader = DataLoader(
        test_data_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=test_sampler,
        num_workers=num_workers,
        persistent_workers=True,
    )

    if args.unknown_prefix is not None or args.cross_data_path is not None:
        if args.cross_data_path is not None:
            cross_set_test = get_costum_dataset(
                data_path=args.cross_data_path,
                ds_type="test",
                only_test_folders=args.only_test_folders,
                only_use=args.cross_sources,
                save_path=save_path,
                limit=args.cross_limit[2],
                asvspoof_name=args.asvspoof_name_cross,
                file_type=args.file_type,
                resample_rate=args.sample_rate,
                seconds=args.seconds,
                get_details=args.get_details,
            )
            cross_set_val = get_costum_dataset(
                data_path=args.cross_data_path,
                ds_type="val",
                only_test_folders=args.only_test_folders,
                only_use=args.cross_sources,
                save_path=save_path,
                limit=args.cross_limit[1],
                asvspoof_name=args.asvspoof_name_cross,
                file_type=args.file_type,
                resample_rate=args.sample_rate,
                seconds=args.seconds,
            )
        else:
            raise NotImplementedError()
            # TODO: remove this.

        if args.ddp:
            cross_val_sampler: DistributedSampler | None = DistributedSampler(
                cross_set_val, shuffle=False, seed=args.seed
            )
            cross_test_sampler: DistributedSampler | None = DistributedSampler(
                cross_set_test, shuffle=False, seed=args.seed
            )
        else:
            cross_val_sampler = cross_test_sampler = None

        cross_loader_val: DataLoader | None = DataLoader(
            cross_set_val,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=cross_val_sampler,
            num_workers=num_workers,
            persistent_workers=True,
        )
        cross_loader_test: DataLoader | None = DataLoader(
            cross_set_test,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=cross_test_sampler,
            num_workers=num_workers,
            persistent_workers=True,
        )
    else:
        cross_loader_val = cross_loader_test = None

    return (
        train_data_loader,
        val_data_loader,
        test_data_loader,
        cross_loader_val,
        cross_loader_test,
    )


class Trainer:
    """Define trainer class for training df detection networks."""

    def __init__(
        self,
        snapshot_path: str,
        args: DotDict,
        normalize: torch.nn.Sequential,
        transforms: torch.nn.Sequential,
        test_data_loader: DataLoader,
        model: torch.nn.Module,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader,
        cross_loader_val: DataLoader,
        cross_loader_test: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fun: torch.nn.Module,
        writer: SummaryWriter | None = None,
    ) -> None:
        """Initialize DDP trainer class.

        Args:
            snapshot_path (str): Path and filename to save the model snapshots at.
            args (DotDict): The training configuration dictionary.
            normalize (torch.nn.Sequential): The normalization transform.
            transforms (torch.nn.Sequential): The frequency-space transform modules.
            test_data_loader (DataLoader): The test data loader to use.
            model (torch.nn.Module): The model to train.
            train_data_loader (DataLoader): The train data loader to use.
            val_data_loader (DataLoader): The validation data loader.
            cross_loader_val (DataLoader): The cross validation data loader (unknown sources).
            cross_loader_test (DataLoader): The cross test data loader.
            optimizer (torch.optim.Optimizer): The optimizer, e.g. Adam.
            loss_fun (torch.nn.Module): The torch loss function module.
            writer (SummaryWriter | None): The tensorboard writer instance. Defaults to None.
        """
        self.args = args

        if self.args.ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
        else:
            self.local_rank = self.global_rank = torch.cuda.current_device()
            self.world_size = 1

        if model is not None:
            self.model = model
            self._check_model_init()
        else:
            self.model = None  # type: ignore

        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.cross_loader_val = cross_loader_val
        self.cross_loader_test = cross_loader_test
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.epochs_run = 0
        self.snapshot_path = snapshot_path + ".pt"
        self.normalize = normalize
        self.transforms = transforms
        self.writer = writer

        self.bar = tqdm([])

        self.validation_list: list = []
        self.loss_list: list = []
        self.accuracy_list: list = []
        self.step_total: int = 0
        self.test_results: tuple = ()
        self.current_true_indices: dict = {}

    def init_model(self, model) -> None:
        """Initialize model.

        This method supports ddp model initialization. Set args.ddp to use distributed training.

        Args:
            model (torch.nn.Module): The model to initialize.

        Raises:
            RuntimeError: If Model is not initialized correctly.
        """
        if model is not None:
            if isinstance(model, torch.nn.Module):
                self.model.to(self.local_rank, non_blocking=True)

                if self.args.ddp:
                    self.model = DiDiP(self.model, device_ids=[self.local_rank])
                return
            elif self.args.ddp and isinstance(model, DiDiP):
                return
            else:
                raise RuntimeError("Given Model not of type torch.nn.Module.")
        else:
            raise RuntimeError("Model to initialize not given.")

    def _check_model_init(self) -> None:
        """Check wether model is initialized correctly.

        Raises:
            RuntimeError: If Model is not initialized correctly.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        else:
            self.init_model(self.model)

        if self.args.ddp and not isinstance(self.model, DiDiP):
            raise RuntimeError("Model not parallelized.")
        else:
            return

    @staticmethod
    def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Return the equal error rate for a binary classifier output.

        Based on:
        https://github.com/scikit-learn/scikit-learn/issues/15247

        Args:
            y_true (np.ndarray): True labels.
            y_score (np.ndarray): Model output.

        Returns:
            float: Equal Error Rate.
        """
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        return eer

    def val_test_loop(
        self,
        data_loader,
        name: str = "",
        pbar: bool = False,
    ) -> tuple[float, float]:
        """Test the performance of a model on a data set by calculating the prediction accuracy and loss of the model.

        Args:
            data_loader (DataLoader): A DataLoader loading the data set on which the performance should be measured,
                e.g. a test or validation set in a data split.
            name (str): Name for tqdm bar (default: "").
            pbar (bool): Disable/Enable tqdm bar (default: False).

        Returns:
            tuple[float, float]: The measured accuracy and eer of the model on the data set.
        """
        ok_sum = 0
        total = 0
        bar = tqdm(
            iter(data_loader), desc="validate", total=len(data_loader), disable=not pbar
        )
        self.model.eval()
        ok_dict = {}
        count_dict = {}
        y_list = []
        out_list = []
        true_indices = torch.tensor([]).to(self.local_rank)
        for val_batch in bar:
            with torch.no_grad():
                freq_time_dt, _ = self.transforms(
                    val_batch["audio"].to(self.local_rank, non_blocking=True)
                )
                freq_time_dt_norm = self.normalize(freq_time_dt)

                out = self.model(freq_time_dt_norm)
                out_max = torch.argmax(out, -1)
                y = val_batch["label"].to(self.local_rank, non_blocking=True) != 0
                ok_mask = out_max == y
                ok_sum += sum(ok_mask).cpu().numpy().astype(int)
                total += len(y)
                bar.set_description(
                    f"[GPU{self.global_rank}] {name} - acc: {ok_sum/total:2.8f}"
                )

                # fill details if custom detailed dataset is used
                if "index" in val_batch.keys():
                    indices = val_batch["index"].to(self.local_rank, non_blocking=True)
                    true_indices = torch.cat(
                        (
                            true_indices,
                            indices[ok_mask == True],  # noqa: E712
                        )
                    ).to(int)

                for lbl, okl in zip(val_batch["label"], ok_mask):
                    lbl_item = lbl.item()
                    if lbl_item not in ok_dict:
                        ok_dict[lbl_item] = [okl.cpu()]
                    else:
                        ok_dict[lbl_item].append(okl.cpu())
                    if lbl_item not in count_dict:
                        count_dict[lbl_item] = 1
                    else:
                        count_dict[lbl_item] += 1

                y_list.append(y)
                out_list.append(out_max)

        common_keys = ok_dict.keys() & count_dict.keys()

        ys = torch.cat(y_list).to(self.local_rank, non_blocking=True)  # type: ignore
        outs = torch.cat(out_list).to(self.local_rank, non_blocking=True)  # type: ignore

        if self.args.ddp:
            # gather ys, outs, ok_sum, total
            ys_gathered = torch.zeros(
                self.world_size * len(ys), dtype=torch.bool, device=self.local_rank
            )  # type: ignore
            outs_gathered = torch.zeros(
                self.world_size * len(outs), dtype=torch.int64, device=self.local_rank
            )  # type: ignore
            ok_sum_gathered: list[Any] = [None for _ in range(self.world_size)]
            total_gathered: list[Any] = [None for _ in range(self.world_size)]
            ok_dict_gathered: list[Any] = [None for _ in range(self.world_size)]
            count_dict_gathered: list[Any] = [None for _ in range(self.world_size)]
            true_indices_gathered: list[Any] = [None for _ in range(self.world_size)]

            torch.distributed.all_gather_into_tensor(ys_gathered, ys)
            torch.distributed.all_gather_into_tensor(outs_gathered, outs)
            torch.distributed.all_gather_object(ok_sum_gathered, ok_sum)
            torch.distributed.all_gather_object(total_gathered, total)
            torch.distributed.all_gather_object(ok_dict_gathered, ok_dict)
            torch.distributed.all_gather_object(count_dict_gathered, count_dict)
            torch.distributed.all_gather_object(true_indices_gathered, true_indices)
        else:
            ys_gathered = ys
            outs_gathered = outs
            true_indices_gathered = true_indices
            ok_sum_gathered = [ok_sum]
            total_gathered = [total]
            ok_dict_gathered = [ok_dict]
            count_dict_gathered = [count_dict]

        if self.args.ddp:
            torch.cuda.synchronize()
            barrier()  # synchronize all processes

        if is_lead(self.args):
            acc_list = Trainer.caculate_acc_dict(
                data_loader, common_keys, ok_dict_gathered, count_dict_gathered
            )

            print(f"{name} - ", acc_list)
            eer = Trainer.calculate_eer(
                ys_gathered.cpu().numpy(), outs_gathered.cpu().numpy()
            )
            val_acc = sum(ok_sum_gathered) / sum(total_gathered)  # type: ignore
            print(
                f"{name} - eer: {eer:2.4f}, Val acc: {val_acc*100:2.2f} %"  # type: ignore
            )
        else:
            eer = 0
            val_acc = 0

        if is_lead(self.args) and isinstance(true_indices_gathered, list):
            true_indices_gathered_cpu = []
            for part in true_indices_gathered:
                true_indices_gathered_cpu.append(part.cpu())
            true_indices_gathered = torch.cat(true_indices_gathered_cpu, dim=0)
        self.current_true_indices[name] = true_indices_gathered

        return val_acc, eer

    @staticmethod
    def caculate_acc_dict(
        data_loader: DataLoader,
        common_keys: set,
        ok_dict_gathered: list,
        count_dict_gathered: list,
    ) -> list[tuple[str | int, float]]:
        """Calculate accuracy over each label using the gathered list.

        Args:
            data_loader (DataLoader): Current dataloader.
            common_keys (set): Set containing the labels to iterate over.
            count_dict_gathered (list): A list for each thread holding a dict of count for occurence of all labels.
            ok_dict_gathered (list): A list for each thread holding a dict of boolean tensors for all correctly
                                     predicted labels.

        Returns:
            list[tuple[str | int, float]]: List of acc for each label, e.g. [("ljspeech", 0.74), ("melgan", 0.81)]
        """
        return [
            (
                data_loader.dataset.get_label_name(key),
                Trainer.calculate_acc_label(count_dict_gathered, ok_dict_gathered, key),
            )
            for key in common_keys
        ]

    @staticmethod
    def calculate_acc_label(
        count_dict_gathered: list, ok_dict_gathered: list, key: int
    ) -> float:
        """Calculate accuracy over one label using the gathered list from each thread.

        See tests in tests/test_trainer.py for examples.

        Args:
            count_dict_gathered (list): A list for each thread holding a dict of count for occurence of all labels.
            ok_dict_gathered (list): A list for each thread holding a dict of boolean tensors for all correctly
                                     predicted labels.
            key (int): Label to look at.

        Returns:
            float: The accuracy for the given label (key) over all threads.

        Raises:
            KeyError: If the requested label is not part of both given lists.
            TypeError: If type of result is not tensor or float.
        """
        all_keys = set()
        for count_dict in count_dict_gathered:
            all_keys.update(count_dict.keys())
        for ok_dict in ok_dict_gathered:
            for key_list in ok_dict.values():
                all_keys.update(key_list)
        all_keys = all_keys.intersection(*[set(d.keys()) for d in count_dict_gathered])
        all_keys = list(  # type: ignore
            all_keys.intersection(*[set(d.keys()) for d in ok_dict_gathered])
        )

        if key not in all_keys:
            raise KeyError(
                f"Key {key} does not exist in both dictionaries. Only available keys: {all_keys}."
            )

        acc = sum(
            [sum(ok_dict_g[key]) for ok_dict_g in ok_dict_gathered]
        ) / sum(  # type: ignore
            [count_dict_g[key] for count_dict_g in count_dict_gathered]
        )  # type: ignore

        if isinstance(acc, torch.Tensor):
            return acc.item()
        elif isinstance(acc, float):
            return acc
        else:
            raise TypeError("Result should either be float or tensor.")

    def integrated_grad(
        self,
        baseline: torch.Tensor,
        image: torch.Tensor,
        target_class_idx: torch.Tensor,
        m_steps: int = 50,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Calculate integrated gradients for a set of images.

        This method and its sub methods are our pytorch version of the implementation at commit 15f45f8 from
        https://github.com/tensorflow/docs/blob/master/site/en/tutorials/interpretability/integrated_gradients.ipynb

        Args:
            baseline (torch.Tensor): Baseline black image.
            image (torch.Tensor): Actual image.
            target_class_idx (torch.Tensor): Image label (target).
            m_steps (int): Number of steps to go. The more, the better
                           the integral approximation. Defaults to 50.
            batch_size (int): Integration batch size. Defaults to 32.

        Returns:
            torch.Tensor: Integrated gradients.
        """
        # Generate alphas.
        alphas = torch.linspace(start=0.0, end=1.0, steps=m_steps + 1).to(
            self.local_rank, non_blocking=True
        )  # type: ignore

        # Collect gradients.
        gradient_batches = []

        # Iterate alphas range and batch computation for speed, memory efficiency, and scaling to larger m_steps.
        for from_ in range(0, len(alphas), batch_size):
            to = min(from_ + batch_size, len(alphas))
            alpha_batch = alphas[from_:to]

            gradient_batch = self.one_batch(
                baseline, image, alpha_batch, target_class_idx
            )
            gradient_batches.append(gradient_batch)

        # Concatenate path gradients together row-wise into single tensor.
        total_gradients = torch.cat(gradient_batches, dim=0)

        # Integral approximation through averaging gradients.
        avg_gradients = integral_approximation(gradients=total_gradients)

        # Scale integrated gradients with respect to input.
        integrated_gradients = (image - baseline) * avg_gradients

        return integrated_gradients

    def one_batch(
        self,
        baseline: torch.Tensor,
        image: torch.Tensor,
        alpha_batch: torch.Tensor,
        target_class_idx: torch.Tensor,
    ) -> torch.Tensor | Any:
        """Interpolate and calculate gradients for one batch of images.

        Args:
            baseline (torch.Tensor): Baseline image.
            image (torch.Tensor): Acutal images.
            alpha_batch (torch.Tensor): Alphas to use.
            target_class_idx (torch.Tensor): Image labels.

        Returns:
            torch.Tensor: Batch of gradients.
        """
        # Generate interpolated inputs between baseline and input.
        interpolated_path_input_batch = interpolate_images(
            baseline=baseline, image=image, alphas=alpha_batch
        )

        # Compute gradients between model outputs and interpolated inputs.
        gradient_batch = self.compute_gradients(
            images=interpolated_path_input_batch, target_class_idx=target_class_idx
        )
        return gradient_batch

    def compute_gradients(
        self,
        images: torch.Tensor,
        target_class_idx: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute gradients for the images using the model classification.

        Args:
            images (torch.Tensor): Images to compute the gradients for.
            target_class_idx (torch.Tensor): The labels corresponding to the images.

        Returns:
            torch.Tensor: Image gradients.
        """
        images.requires_grad = True
        logits = self.model(images)
        probs = torch.nn.functional.softmax(logits, dim=-1)[:, target_class_idx]
        probs.backward(torch.ones_like(probs))
        return images.grad

    def integrated_gradients(
        self,
        model_file: str = "ig",
        pbar: bool = True,
    ) -> None:
        """Calculate and save integrated gradients.

        Args:
            model_file (str): Prefix for the save path. Defaults to "ig".
            pbar (bool): Disable/Enable tqdm bar. Defaults to True.
        """
        plot_path = self.args.log_dir + "/plots/"

        if not os.path.exists(plot_path):
            os.mkdir(plot_path)

        welford_ig = Mean()
        welford_sal = Mean()

        data_loader = self.cross_loader_test
        bar = tqdm(
            iter(data_loader),  # type: ignore
            desc="integrate grads",
            total=len(data_loader),  # type: ignore
            disable=not pbar,
        )

        index = 0
        index_0 = 0
        index_1 = 0
        both = False
        if self.args.target is None:
            both = True
            target_value = 1
        else:
            try:
                target_value = int(self.args.target)
            except ValueError:
                target_value = 1

        target = torch.tensor(target_value).to(
            f"cuda:{str(self.local_rank)}", non_blocking=True
        )

        if self.args.ig_times_per_target is not None:
            times = times_0 = times_1 = self.args.ig_times_per_target
        else:
            times = times_0 = times_1 = 2500

        target_0 = torch.tensor(0).to(f"cuda:{str(self.local_rank)}", non_blocking=True)
        target_1 = torch.tensor(1).to(f"cuda:{str(self.local_rank)}", non_blocking=True)
        batch_size = 128
        m_steps = 200

        self.model.zero_grad()

        for val_batch in bar:
            label = (
                val_batch["label"].to(f"cuda:{str(self.local_rank)}", non_blocking=True)
                != 0
            ).type(torch.long)
            if label.shape[0] != batch_size:
                continue

            label[label > 0] = 1
            if not both and target not in label:
                continue
            elif (
                both
                and index_0 == times_0
                and index_1 != times_1
                and target_1 not in label
            ):
                continue
            elif (
                both
                and index_1 == times_1
                and index_0 != times_0
                and target_0 not in label
            ):
                continue

            freq_time_dt, _ = self.transforms(
                val_batch["audio"].to(self.local_rank, non_blocking=True)
            )
            freq_time_dt_norm = self.normalize(freq_time_dt)

            baseline = torch.zeros_like(freq_time_dt_norm[0]).to(
                f"cuda:{str(self.local_rank)}", non_blocking=True
            )
            for i in tqdm(range(freq_time_dt_norm.shape[0])):
                image = freq_time_dt_norm[i]
                c_label = label[i]
                if not both and c_label != target:
                    continue
                elif (
                    both
                    and c_label == target_0
                    and index_0 == times_0
                    and index_1 != times_1
                ):
                    continue
                elif (
                    both
                    and c_label == target_1
                    and index_1 == times_1
                    and index_0 != times_0
                ):
                    continue
                elif both and index_0 == times_0 and index_1 == times_1:
                    break
                elif not both and index == times:
                    break

                attributions = self.integrated_grad(
                    baseline=baseline,
                    image=image,
                    target_class_idx=c_label,
                    m_steps=m_steps,
                )

                attribution_mask = torch.sum(attributions, dim=0).unsqueeze(0)
                welford_ig.update(attribution_mask)
                welford_sal.update(image)

                if c_label == target_0:
                    index_0 += 1
                elif c_label == target_1:
                    index_1 += 1
                index += 1

            torch.cuda.empty_cache()
            if both and index_0 == times_0 and index_1 == times_1:
                break
            elif not both and index == times:
                break

        with torch.no_grad():
            print("index 0 ", index_0)
            print("index 1 ", index_1)
            print("index ", index)
            mean_ig = welford_ig.finalize()
            mean_sal = welford_sal.finalize()

            if is_lead(self.args):
                if both:
                    target_str = "01"
                else:
                    target_str = str(target.detach().cpu().item())

                path = (
                    plot_path
                    + model_file.replace("/", "_")
                    + "_"
                    + "-".join(self.args.cross_sources)
                    + f"x{times}_target-{target_str}"
                )
                np.save(
                    path + "_integrated_gradients.npy",
                    mean_ig.detach().cpu().numpy(),
                )
                np.save(
                    path + "_mean_images.npy", mean_sal.squeeze().detach().cpu().numpy()
                )
                np.save(
                    path + "_last_image.npy", image.squeeze().detach().cpu().numpy()
                )

    def _run_test(
        self, only_unknown: bool = False
    ) -> tuple[float, float, float, float]:
        """Run tests on all test data loaders.

        Args:
            only_unknown (bool): If True only cross_data_loaders will be tested/used. Defaults to False.

        Returns:
            tuple[float, float, float, float]: Test accuracy, Test EER, Cross Test Acc., Cross Test EER
        """
        self.model.eval()
        with torch.no_grad():
            if not only_unknown:
                test_acc, test_eer = self.val_test_loop(
                    data_loader=self.test_data_loader,
                    pbar=self.args.pbar,
                    name="test known",
                )
            else:
                test_acc = test_eer = 0
            if (
                self.args.unknown_prefix is not None
                or self.args.cross_data_path is not None
            ):
                cr_test_acc, cr_test_eer = self.val_test_loop(
                    data_loader=self.cross_loader_test,
                    pbar=self.args.pbar,
                    name="test unknown",
                )
            else:
                cr_test_eer = cr_test_acc = 0

        if self.args.tensorboard and is_lead(self.args):
            self.writer.add_scalar("accuracy/test", test_acc, self.step_total)  # type: ignore
            self.writer.add_scalar("eer/test", test_eer, self.step_total)  # type: ignore
            self.writer.add_scalar("accuracy/cross_test", cr_test_acc, self.step_total)  # type: ignore
            self.writer.add_scalar("eer/cross_test", cr_test_eer, self.step_total)  # type: ignore

        return test_acc, test_eer, cr_test_acc, cr_test_eer

    def _run_epoch(self, epoch: int) -> None:
        """Iterate over training data.

        Args:
            epoch (int): The epoch to run.
        """
        self.bar = tqdm(
            iter(self.train_data_loader),  # type: ignore
            desc="training cnn",
            total=len(self.train_data_loader),  # type: ignore
            unit="batches",
            disable=not self.args.pbar,
        )
        if is_lead(self.args):
            print(
                f"+------------------- Epoch {epoch+1} -------------------+", flush=True
            )
            if self.args.tensorboard:
                self.writer.add_scalar("epochs", epoch, self.step_total)  # type: ignore
        if self.args.ddp:
            self.train_data_loader.sampler.set_epoch(epoch)  # type: ignore

        batch: torch.Tensor
        for _it, batch in enumerate(self.bar):
            self.model.train()
            self._run_batch(epoch, batch)

    def _run_validation(self, epoch: int) -> None:
        """Iterate over validation data.

        Args:
            epoch (int): Current epoch.
        """
        val_acc, val_eer = self.val_test_loop(
            data_loader=self.val_data_loader,
            pbar=self.args.pbar,
            name="val known",
        )

        if (
            self.args.unknown_prefix is not None
            or self.args.cross_data_path is not None
        ):
            cr_val_acc, cr_val_eer = self.val_test_loop(
                data_loader=self.cross_loader_val,
                pbar=self.args.pbar,
                name="val unknown",
            )

        if self.args.tensorboard and is_lead(self.args):
            self.writer.add_scalar("accuracy/validation", val_acc, self.step_total)  # type: ignore
            self.writer.add_scalar("eer/validation", val_eer, self.step_total)  # type: ignore
            self.writer.add_scalar(  # type: ignore
                "accuracy/cross_validation", cr_val_acc, self.step_total
            )
            self.writer.add_scalar("eer/cross_validation", cr_val_eer, self.step_total)  # type: ignore
            self.writer.add_scalar("epochs", epoch, self.step_total)  # type: ignore

    def _run_batch(self, e: int, batch: torch.Tensor) -> None:
        """Run one batch iteration forward pass.

        Args:
            e (int): Current epoch.
            batch (torch.Tensor): Current batch to run.
        """
        batch_audios = batch[self.train_data_loader.dataset.key].to(
            self.local_rank, non_blocking=True
        )
        batch_labels = (
            batch["label"].to(self.local_rank, non_blocking=True) != 0
        ).type(torch.long)

        if self.args.aug_contrast:
            batch_audios = contrast(batch_audios)
        if self.args.aug_noise:
            batch_audios = add_noise(batch_audios)

        self.optimizer.zero_grad()
        with torch.no_grad():
            freq_time_dt, _ = self.transforms(batch_audios)
            freq_time_dt_norm = self.normalize(freq_time_dt)

        out = self.model(freq_time_dt_norm)
        loss = self.loss_fun(out, batch_labels)
        acc = (
            torch.sum(
                (
                    torch.argmax(out, -1)
                    == (batch_labels != 0).to(self.local_rank, non_blocking=True)
                )
            )
            / self.args.batch_size
        )

        self.bar.set_description(
            f"[GPU{self.global_rank}] ce-cost: {loss.item():2.8f}, acc: {acc.item():2.2f}"
        )

        loss.backward()
        self.optimizer.step()
        self.step_total += 1
        self.loss_list.append([self.step_total, e, loss.item()])
        self.accuracy_list.append([self.step_total, e, acc.item()])

        if self.args.tensorboard and is_lead(self.args):
            self.writer.add_scalar("loss/train", loss.item(), self.step_total)  # type: ignore
            self.writer.add_scalar("accuracy/train", acc.item(), self.step_total)  # type: ignore
            if self.step_total == 0:
                self.writer.add_graph(self.model, batch_audios)  # type: ignore

    def _save_snapshot(self, epoch: int) -> None:
        """Save snapshot of current model.

        Args:
            epoch (int): Current epoch.
        """
        snapshot = {
            "MODEL_STATE": self.model.state_dict() if self.args.ddp else self.model.state_dict(),  # type: ignore
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch+1} | Training snapshot saved at {self.snapshot_path}")

    def load_snapshot(self, snapshot_path: str) -> None:
        """Load snapshot from given path.

        Args:
            snapshot_path (str): Snapshot path to load model from.
        """
        loc = {"cuda:%d" % 0: "cuda:%d" % self.local_rank}
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

    def train(self, max_epochs: int) -> None:
        """Train model on a given number of epochs.

        Args:
            max_epochs (int): Number of epochs to train the model.
        """
        self._check_model_init()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            if is_lead(self.args):
                if (
                    (epoch > 0 and epoch % self.args.ckpt_every == 0)
                    or (epoch == 0 and self.args.ckpt_every == 1)
                    or (epoch == max_epochs)
                ):
                    self._save_snapshot(epoch)
            if (epoch > 0 and epoch % self.args.validation_interval == 0) or (
                epoch == 0 and self.args.validation_interval == 1
            ):
                self._run_validation(epoch)
            if epoch == max_epochs - 1:
                if is_lead(self.args):
                    print("Training done, now testing...")
                test_results = self.testing()
                self.test_results = test_results
                if is_lead(self.args):
                    print(
                        f"test results: known acc {test_results[0]*100:2.2f} %, \
                        known eer {test_results[1]:.3f}, \
                        unknown acc {test_results[2]*100:2.2f} %, \
                        unknown eer {test_results[3]:.3f}"
                    )

    def testing(self, only_unknown: bool = False) -> tuple[float, float, float, float]:
        """Check model initialization and run all tests.

        Args:
            only_unknown (bool): If True only cross_data_loaders will be tested/used. Defaults to False.

        Returns:
            tuple[float, float, float, float]: Test accuracy, Test EER, Cross Test Acc., Cross Test EER
        """
        self._check_model_init()
        return self._run_test(only_unknown=only_unknown)


def is_lead(args: DotDict) -> bool:
    """Check if current process is lead rank.

    Args:
        args (DotDict): Experiment configuration dictionary.

    Returns:
        bool: True if DDP is off or local rank is 0.
    """
    if not args.ddp:
        return True
    elif int(os.environ["LOCAL_RANK"]) == 0 and int(os.environ["RANK"]) == 0:
        return True
    return False


def main() -> None:
    """Trains a model to classify audios.

    All settings such as which model to use, parameters, normalization, data set path,
    seed etc. are specified via cmd line args.
    All training, validation and testing results are printed to stdout.
    After the training is done, the results are stored in a pickle dump in the 'log' folder.
    The state_dict of the trained model is stored there as well.

    Raises:
        NotImplementedError: If wrong features are combined with the wrong model.
        ValueError: If stft is started with signed log scaling.
        TypeError: If there went something wrong with the results.
        RuntimeError: If config path is not given even though gridsearch is enabled.
    """
    torch.set_num_threads(24)  # adjust this according to the number of cores available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    parsed_args = _parse_args()
    args = DotDict(vars(parsed_args))

    args.num_workers = 10  # adjust this if you have data loading problems

    if args.ddp:
        ddp_setup()

    if is_lead(args):
        print(parsed_args)

    base_dir = args.log_dir
    if not os.path.exists(base_dir + "/models"):
        os.makedirs(base_dir + "/models")
    if not os.path.exists(base_dir + "/tensorboard"):
        os.makedirs(base_dir + "/tensorboard")
    if not os.path.exists(base_dir + "/norms"):
        os.makedirs(base_dir + "/norms")

    num_exp = 1
    exp_results: dict[str, Any] = {}
    if args.enable_gs:
        if is_lead(args):
            print("--------------- Starting grid search -----------------")

        if args.config is None or args.config == "":
            raise RuntimeError("Config file must be provided.")

        config_namespace = {}  # type: ignore

        with open(args.config, "r") as config_script:
            exec(config_script.read(), config_namespace)  # noqa: S102

        get_config = config_namespace.get("get_config")

        config = {}
        if get_config:
            config = get_config()

        griderator = build_new_grid(
            config, random_seeds=args.random_seeds, seeds=args.init_seeds
        )
        num_exp = griderator.get_len()

    for _exp_number in range(num_exp):
        if args.enable_gs:
            if is_lead(args):
                print("---------------------------------------------------------")
                print(
                    f"starting new experiments with {griderator.grid_values[griderator.current]}"
                )
                print("---------------------------------------------------------")
            args, _ = griderator.update_step(args)

        if args.features != "none" and args.model != "lcnn":
            raise NotImplementedError(
                f"LFCC features are currently not implemented for {args.model}."
            )

        path_name = args.data_prefix.split("/")[-1].split("_")

        transform = args.transform
        features = args.features
        known_gen_name = path_name[4]
        loss_less = False if args.loss_less == "False" else True

        if transform == "stft" and loss_less:
            raise ValueError(
                "Sign channel not possible for stft due to complex data type."
            )

        # fix the seed in the interest of reproducible results.
        set_seed(args.seed)

        transforms, normalize = get_transforms(
            args,
            features,
            device,
            args.calc_normalization,
            pbar=args.pbar,
        )

        args.input_dim = get_input_dims(args=args, transforms=transforms)

        try:
            model = get_model(
                args=args,
                model_name=args.model,
                nclasses=args.nclasses,
                in_channels=2 if loss_less else 1,
                lead=is_lead(args),
            )
        except RuntimeError:
            print("Skipping model args.model_conf")
            continue

        model_name = model.get_name() if args.model == "modules" else "customModel"

        (
            train_data_loader,
            val_data_loader,
            test_data_loader,
            cross_loader_val,
            cross_loader_test,
        ) = create_data_loaders(
            args=args,
            num_workers=args.num_workers,
        )

        loss_fun = torch.nn.CrossEntropyLoss()

        lr = args.learning_rate
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            weight_decay=args.weight_decay,
        )

        model_file = base_dir + "/models/" + path_name[0] + "_"
        if transform == "stft":
            model_file += "stft"
        elif transform == "packets":
            model_file += "packets" + str(args.wavelet)
        model_file += (
            "_"
            + str(args.features)
            + "_"
            + str(args.hop_length)
            + "_"
            + str(args.sample_rate)
            + "_"
            + str(args.window_size)
            + "_"
            + str(args.num_of_scales)
            + "_"
            + str(int(args.f_min))
            + "-"
            + str(int(args.f_max))
            + "_"
            + path_name[3]
            + "_"
            + str(args.learning_rate)
            + "_"
            + str(args.weight_decay)
            + "_"
            + str(args.batch_size)
            + "_"
            + str(args.nclasses)
            + "_"
            + f"{args.epochs}e"
            + "_"
            + str(model_name)
            + "_signs"
            + str(loss_less)
            + "_augc"
            + str(args.aug_contrast)
            + "_augn"
            + str(args.aug_noise)
            + "_power"
            + str(args.power)
            + "_"
            + str(args.only_use[1])
            + "_"
            + str(args.seconds)
            + "secs_"
            + str(args.seed)
        )

        if args.tensorboard and is_lead(args):
            writer_str = base_dir + "/tensorboard/"
            writer_str += f"{model_name}/"
            writer_str += f"{args.transform}/"
            if transform == "packets":
                writer_str += f"{args.wavelet}/"
            writer_str += f"{args.features}/"
            writer_str += f"{args.batch_size}_"
            writer_str += f"{args.learning_rate}_"
            writer_str += f"{args.weight_decay}_"
            writer_str += f"{args.epochs}/"
            writer_str += f"{args.f_min}-"
            writer_str += f"{args.f_max}/"
            writer_str += f"{args.num_of_scales}/"
            writer_str += f"signs{loss_less}/"
            writer_str += f"augc{args.aug_contrast}/"
            writer_str += f"augn{args.aug_noise}/"
            writer_str += f"power{args.power}/"
            writer_str += f"{known_gen_name}/"
            writer_str += f"{args.seed}"
            writer = SummaryWriter(writer_str, max_queue=100)
        else:
            writer = None  # type: ignore

        trainer = Trainer(
            model=model,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            test_data_loader=test_data_loader,
            cross_loader_val=cross_loader_val,
            cross_loader_test=cross_loader_test,
            optimizer=optimizer,
            loss_fun=loss_fun,
            normalize=normalize,
            transforms=transforms,
            snapshot_path=model_file,
            args=args,
            writer=writer,
        )

        if args.only_testing:
            trainer._check_model_init()
            trainer.load_snapshot(trainer.snapshot_path)
            trainer.test_results = trainer.testing(only_unknown=True)
        elif args.only_ig:
            trainer._check_model_init()
            print("loading " + trainer.snapshot_path)
            trainer.load_snapshot(trainer.snapshot_path)
            path = f"{args.transform}_{args.sample_rate}_{args.seconds}"
            path += f"_{args.seed}_{args.only_use[-1]}_{args.wavelet}_{args.power}_{str(loss_less)}"
            trainer.integrated_gradients(path)
        else:
            trainer.train(args.epochs)

        if exp_results.get(args.seed) is None:
            exp_results[args.seed] = [trainer.test_results]
        elif type(exp_results[args.seed]) is list:
            exp_results[args.seed].append(trainer.test_results)
        else:
            raise TypeError("Result array must contain lists.")

        known_indices = trainer.current_true_indices.get("test known", [])
        unknown_indices = trainer.current_true_indices.get("test unknown", [])
        if args.get_details and (len(known_indices) > 0 or len(unknown_indices) > 0):
            if isinstance(known_indices, torch.Tensor):
                true_ind_known = known_indices.detach().cpu().numpy()
            elif isinstance(known_indices, np.ndarray):
                true_ind_known = known_indices
            else:
                true_ind_known = []
            if isinstance(unknown_indices, torch.Tensor):
                true_ind_unknown = unknown_indices.detach().cpu().numpy()
            elif isinstance(unknown_indices, np.ndarray):
                true_ind_unknown = unknown_indices
            else:
                true_ind_unknown = []

            true_ind_data = {
                "known": true_ind_known,
                "unknown": true_ind_unknown,
                "dataset": trainer.cross_loader_test.dataset.audio_data,
            }
            np.save(
                f"{args.log_dir}/true_ind_{model_file.split('/')[-1]}_{args.seed}.npy",
                true_ind_data,
            )

    if is_lead(args):
        if (
            args.tensorboard
            and writer is not None
            and isinstance(writer, SummaryWriter)
        ):
            writer.close()
        print_results(args, exp_results, griderator, model_file)

    if args.ddp:
        destroy_process_group()


def print_results(
    args: DotDict,
    exp_results: dict,
    griderator: _Griderator,
    model_file: str = "defaultmodel",
) -> None:
    """Print results of all experiments.

    Args:
        args (DotDict): Experiment configuration.
        exp_results (dict): Experiment results.
        griderator (_Griderator): Experiment list wrapper class.
        model_file (str): Model file string (unique for this experiment config).
    """
    results = np.asarray(list(exp_results.values()))
    if results.shape[0] == 0:
        exit(0)

    if args.transform == "packets":
        if griderator.init_config and "wavelet" in griderator.init_config:
            wavelets = griderator.init_config["wavelet"]
        elif hasattr(args, "wavelet"):
            wavelets = [args.wavelet]
        else:
            wavelets = ["default"]
    else:
        wavelets = ["stft"]

    np.save(
        args.log_dir + f"/{model_file.split('/')[-1]}_{','.join(wavelets)}_results.npy",
        results,
    )
    mean = results.mean(0)
    std = results.std(0)
    print("results:", results)
    print(mean)
    print(std)

    if True:
        print("evaluating results:")
        min = results.min(0)
        max = results.max(0)
        stringer = []
        stringer_2 = []
        for i in range(len(mean)):
            print("------------------------------------------------------------------")
            stringer_2.append(
                {k: v for k, v in zip(griderator.get_keys(), griderator.grid_values[i])}
            )
            output = rf"& ${max[i, 2]*100:.2f}$ & ${mean[i, 2]*100:.2f} \pm {std[i, 2]*100:.2f}$ &"
            output += rf" ${min[i, 3]:.3f}$ & ${mean[i, 3]:.3f} \pm {std[i, 3]:.3f}$ \\"
            stringer.append(output)

        stringer_array = np.asarray(stringer, dtype=object)
        print(stringer_array)
        cross_dirs = griderator.init_config["cross_sources"]
        stringer_array = stringer_array.reshape((len(wavelets), len(cross_dirs)))
        for i in range(len(cross_dirs)):
            print("+---------------------+")
            print(cross_dirs[i])  # which configs
            for k in range(len(wavelets)):
                print(rf"{wavelets[k]} & {stringer_array[k][i]}")  # which values
        print("+---------------------+")
    print("------------------------------------------------------------------")
    print(
        f"Best unknown eer: {mean[np.argmin(mean[:,3]), 3]:.4f} +- {std[np.argmin(mean[:,3]), 3]:.4f}"
    )

    if args.enable_gs:
        best_config = {
            k: v
            for k, v in zip(
                griderator.get_keys(), griderator.grid_values[np.argmin(mean[:, 3])]
            )
        }
        print(f"Best config: {best_config}")


def _parse_args():
    """Parse cmd line args for training an audio classifier."""
    parser = argparse.ArgumentParser(description="Train an audio classifier")
    parser = add_default_parser_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
