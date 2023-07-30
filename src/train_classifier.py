"""Source code to train audio deepfake detectors in wavelet space."""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DiDiP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.data_loader import CrossWavefakeDataset, LearnWavefakeDataset
from src.models import get_model, save_model
from src.utils import (
    DotDict,
    add_default_parser_args,
    add_noise,
    contrast,
    init_grid,
    set_seed,
)
from src.wavelet_math import get_transforms


def ddp_setup() -> None:
    """Initialize distributed data parallel environment."""
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def create_data_loaders(
    args,
    limit: int = -1,
    num_workers: int = 5,
) -> tuple:
    """Create the data loaders needed for training.

    The test set is created outside a loader.

    Args:
        data_prefix (str): Where to look for the data.
        batch_size (int): preferred training batch size.

    Returns:
        dataloaders (tuple): train_data_loader, val_data_loader, test_data_set
    """
    train_data_set = LearnWavefakeDataset(args.data_prefix + "_train", limit=limit)
    val_data_set = LearnWavefakeDataset(args.data_prefix + "_val", limit=limit)
    test_data_set = LearnWavefakeDataset(args.data_prefix + "_test", limit=limit)

    if args.ddp:
        train_sampler = DistributedSampler(
            train_data_set, shuffle=True, seed=args.seed, drop_last=True
        )
        val_sampler = DistributedSampler(val_data_set, shuffle=False, seed=args.seed)
        test_sampler = DistributedSampler(test_data_set, shuffle=False, seed=args.seed)
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

    if args.unknown_prefix is not None or args.cross_dir is not None:
        if args.cross_dir is not None:
            cross_set_test = CrossWavefakeDataset(
                base_path=args.cross_dir,
                prefix=args.cross_prefix,
                sources=args.cross_sources,
                limit=1000,
            )
            cross_set_val = CrossWavefakeDataset(
                base_path=args.cross_dir,
                prefix=args.cross_prefix,
                sources=args.cross_sources,
                limit=2000,
            )
        else:
            cross_set_val = LearnWavefakeDataset(
                args.unknown_prefix + "_val",
                limit=1000,
            )
            cross_set_test = LearnWavefakeDataset(
                args.unknown_prefix + "_test",
                limit=2000,
            )

        if args.ddp:
            cross_val_sampler = DistributedSampler(
                cross_set_val, shuffle=False, seed=args.seed
            )
            cross_test_sampler = DistributedSampler(
                cross_set_test, shuffle=False, seed=args.seed
            )
        else:
            cross_val_sampler = cross_test_sampler = None

        cross_loader_val = DataLoader(
            cross_set_val,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            sampler=cross_val_sampler,
            num_workers=num_workers,
            persistent_workers=True,
        )
        cross_loader_test = DataLoader(
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
        args,
        normalize,
        transforms,
        test_data_loader: DataLoader,
        model: torch.nn.Module | None = None,
        train_data_loader: DataLoader | None = None,
        val_data_loader: DataLoader | None = None,
        cross_loader_val: DataLoader | None = None,
        cross_loader_test: DataLoader | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fun=None,
        writer: SummaryWriter | None = None,
    ) -> None:
        """Initialize trainer."""
        self.args = args

        if self.args.ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ['WORLD_SIZE'])
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

        self.bar = None

        self.validation_list: list = []
        self.loss_list: list = []
        self.accuracy_list: list = []
        self.step_total: int = 0
        self.test_results: tuple = ()

    def init_model(self, model) -> None:
        """Initialize model.

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

    def calculate_eer(self, y_true, y_score) -> float:
        """Return the equal error rate for a binary classifier output.

        Based on:
        https://github.com/scikit-learn/scikit-learn/issues/15247
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
            Tuple[float, Any]: The measured accuracy and eer of the model on the data set.
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
                for lbl, okl in zip(val_batch["label"], ok_mask):
                    if lbl.item() not in ok_dict:
                        ok_dict[lbl.item()] = [okl.cpu()]
                    else:
                        ok_dict[lbl.item()].append(okl.cpu())
                    if lbl.item() not in count_dict:
                        count_dict[lbl.item()] = 1
                    else:
                        count_dict[lbl.item()] += 1

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
            ok_sum_gathered = [None for _ in range(self.world_size)]  # type: ignore
            total_gathered = [None for _ in range(self.world_size)]  # type: ignore
            ok_dict_gathered = [None for _ in range(self.world_size)]  # type: ignore
            count_dict_gathered = [None for _ in range(self.world_size)]  # type: ignore

            torch.distributed.all_gather_into_tensor(ys_gathered, ys)
            torch.distributed.all_gather_into_tensor(outs_gathered, outs)
            torch.distributed.all_gather_object(ok_sum_gathered, ok_sum)
            torch.distributed.all_gather_object(total_gathered, total)
            torch.distributed.all_gather_object(ok_dict_gathered, ok_dict)
            torch.distributed.all_gather_object(count_dict_gathered, count_dict)
        else:
            ys_gathered = ys
            outs_gathered = outs
            ok_sum_gathered = [ok_sum]
            total_gathered = [total]
            ok_dict_gathered = [ok_dict]
            count_dict_gathered = [count_dict]

        # torch.cuda.synchronize()
        if is_lead(self.args):
            print(
                f"{name} - ",
                [
                    (
                        data_loader.dataset.get_label_name(key),
                        (
                            sum([sum(ok_dict_g[key]) for ok_dict_g in ok_dict_gathered])  # type: ignore
                            / sum(
                                [
                                    count_dict_g[key]  # type: ignore
                                    for count_dict_g in count_dict_gathered
                                ]  # type: ignore
                            )
                        ).item(),
                    )
                    for key in common_keys
                ],
            )  # type: ignore
            eer = self.calculate_eer(
                ys_gathered.cpu().numpy(), outs_gathered.cpu().numpy()
            )
            val_acc = sum(ok_sum_gathered) / sum(total_gathered)  # type: ignore
            print(
                f"{name} - eer: {eer:2.4f}, Val acc: {val_acc*100:2.2f} %"  # type: ignore
            )
        else:
            eer = 0
            val_acc = 0
        return val_acc, eer  # type: ignore

    def _run_test(self) -> tuple[float, float, float, float]:
        self.model.eval()
        with torch.no_grad():
            test_acc, test_eer = self.val_test_loop(
                data_loader=self.test_data_loader,
                pbar=self.args.pbar,
                name="test known",
            )
            if self.args.unknown_prefix is not None or self.args.cross_dir is not None:
                cr_test_acc, cr_test_eer = self.val_test_loop(
                    data_loader=self.cross_loader_test,
                    pbar=self.args.pbar,
                    name="test unknown",
                )
            else:
                cr_test_eer = cr_test_acc = 0

        if self.args.tensorboard:
            self.writer.add_scalar("accuracy/test", test_acc, self.step_total)  # type: ignore
            self.writer.add_scalar("eer/test", test_eer, self.step_total)  # type: ignore
            self.writer.add_scalar("accuracy/cross_test", cr_test_acc, self.step_total)  # type: ignore
            self.writer.add_scalar("eer/cross_test", cr_test_eer, self.step_total)  # type: ignore

        return test_acc, test_eer, cr_test_acc, cr_test_eer

    def _run_epoch(self, e):
        """Iterate over training data."""
        self.bar = tqdm(
            iter(self.train_data_loader),
            desc="training cnn",
            total=len(self.train_data_loader),
            unit="batches",
            disable=not self.args.pbar,
        )
        if is_lead(self.args):
            print(f"+------------------- Epoch {e+1} -------------------+", flush=True)
        if self.args.ddp:
            self.train_data_loader.sampler.set_epoch(e)

        for _it, batch in enumerate(self.bar):
            self.model.train()
            self._run_batch(e, batch)

    def _run_validation(self, epoch):
        """Iterate over validation data."""
        val_acc, val_eer = self.val_test_loop(
            data_loader=self.val_data_loader,
            pbar=self.args.pbar,
            name="val known",
        )

        if self.args.unknown_prefix is not None or self.args.cross_dir is not None:
            cr_val_acc, cr_val_eer = self.val_test_loop(
                data_loader=self.cross_loader_val,
                pbar=self.args.pbar,
                name="val unknown",
            )

        if self.args.tensorboard:
            self.writer.add_scalar("accuracy/validation", val_acc, self.step_total)
            self.writer.add_scalar("eer/validation", val_eer, self.step_total)
            self.writer.add_scalar(
                "accuracy/cross_validation", cr_val_acc, self.step_total
            )
            self.writer.add_scalar("eer/cross_validation", cr_val_eer, self.step_total)

        if self.args.tensorboard:
            self.writer.add_scalar("epochs", epoch, self.step_total)

    def _run_batch(self, e, batch):
        """Run one batch iteration forward pass."""
        batch_audios = batch[self.train_data_loader.dataset.key].to(
            self.local_rank, non_blocking=True
        )
        batch_labels = (
            batch["label"].to(self.local_rank, non_blocking=True) != 0
        ).type(torch.long)

        if self.args.aug_contrast:
            batch_audios = contrast(batch_audios)
            # batch_audios = torchaudio.functional.preemphasis(batch_audios, np.random.uniform(0., 1.0))
            # batch_audios = torchaudio.functional.dither(batch_audios)
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

        if self.args.tensorboard:
            self.writer.add_scalar("loss/train", loss.item(), self.step_total)
            self.writer.add_scalar("accuracy/train", acc.item(), self.step_total)
            if self.step_total == 0:
                self.writer.add_graph(self.model, batch_audios)

    def _save_snapshot(self, epoch) -> None:
        """Save snapshot of current model."""
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if self.args.ddp else self.model.state_dict(),  # type: ignore
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch+1} | Training snapshot saved at {self.snapshot_path}")

    def load_snapshot(self, snapshot_path):
        """Load snapshot from given path."""
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]

    def train(self, max_epochs: int) -> None:
        """Train model."""
        self._check_model_init()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

            if is_lead(self.args):
                if epoch % self.args.ckpt_every == 0:
                    self._save_snapshot(epoch)
            if epoch % self.args.validation_interval == 0:
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

    def testing(self) -> tuple[float, float, float, float]:
        """Iterate over test set."""
        self._check_model_init()
        return self._run_test()


def is_lead(args) -> bool:
    """Check if current process is lead rank."""
    if not args.ddp:
        return True
    elif int(os.environ["LOCAL_RANK"]) == 0 and int(os.environ["RANK"]) == 0:
        return True
    return False


def main():
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
    """
    torch.set_num_threads(24)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.multiprocessing.set_start_method("spawn")

    parsed_args = _parse_args()
    args = DotDict(vars(parsed_args))

    args.num_workers = 8

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
    exp_results = {}
    if args.enable_gs:
        if is_lead(args):
            print("--------------- Starting grid search -----------------")

        if not args.random_seeds:
            griderator = init_grid(num_exp=5, init_seeds=[0, 1, 2, 3, 4])
        else:
            griderator = init_grid(num_exp=3)
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

        if args.f_max > args.sample_rate / 2:
            print("Warning: maximum analyzed frequency is above nyquist rate.")

        if args.features != "none" and args.model != "lcnn":
            raise NotImplementedError(
                f"LFCC features are currently not implemented for {args.model}."
            )

        path_name = args.data_prefix.split("/")[-1].split("_")

        transform = args.transform
        features = args.features
        known_gen_name = path_name[4]
        loss_less = False if args.loss_less == "False" else True

        if args.model == "onednet" and loss_less:
            raise NotImplementedError(
                "OneDNet does not work together with the sign channel."
            )

        if transform == "stft" and loss_less:
            raise ValueError(
                "Sign channel not possible for stft due to complex data type."
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
            + str(args.model)
            + "_signs"
            + str(loss_less)
            + "_augc"
            + str(args.aug_contrast)
            + "_augn"
            + str(args.aug_noise)
            + "_power"
            + str(args.power)
            + "_"
            + known_gen_name
            + "_"
            + str(args.seed)
        )

        if args.tensorboard:
            writer_str = base_dir + "/tensorboard/"
            writer_str += f"{args.model}/"
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

        # fix the seed in the interest of reproducible results.
        set_seed(args.seed)

        if "doubledelta" in features:
            channels = 60
        elif "delta" in features:
            channels = 40
        elif "lfcc" in features:
            channels = 20
        else:
            channels = int(args.num_of_scales)

        try:
            model = get_model(
                args=args,
                model_name=args.model,
                nclasses=args.nclasses,
                num_of_scales=args.num_of_scales,
                flattend_size=args.flattend_size,
                in_channels=2 if loss_less else 1,
                channels=channels,
                dropout_cnn=args.dropout_cnn,
                dropout_lstm=args.dropout_lstm,
            )
        except RuntimeError:
            print(f"Skipping model args.model_conf")
            continue

        (
            train_data_loader,
            val_data_loader,
            test_data_loader,
            cross_loader_val,
            cross_loader_test,
        ) = create_data_loaders(
            args=args,
            limit=20000,
            num_workers=args.num_workers,
        )

        loss_fun = torch.nn.CrossEntropyLoss()

        optimizer = Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        transforms, normalize = get_transforms(
            args,
            args.data_prefix,
            features,
            device,
            args.calc_normalization,
            pbar=args.pbar,
        )

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
        trainer.train(args.epochs)

        if exp_results.get(args.seed) is None:
            exp_results[args.seed] = [trainer.test_results]
        elif type(exp_results[args.seed]) is list:
            exp_results[args.seed].append(trainer.test_results)
        else:
            raise TypeError("Result array must contain lists.")

        if args.tensorboard:
            writer.close()

    if is_lead(args):
        results = np.asarray(list(exp_results.values()))
        mean = results.mean(0)
        std = results.std(0)
        print(mean)
        print(
            f"Best unknown eer: {mean[np.argmin(mean[:,3]), 3]:.4f} +- {std[np.argmin(mean[:,3]), 3]:.4f}"
        )

        if args.enable_gs:
            print(griderator.grid_values)
            best_config = {
                k: v
                for k, v in zip(
                    griderator.get_keys(), griderator.grid_values[np.argmin(mean[:, 3])]
                )
            }
            print(f"Best config: {best_config}")

    if args.ddp:
        destroy_process_group()


def save_model_epoch(model_file, model) -> str:
    """Save model each epoch, in case the script aborts for some reason."""
    save_model(model, model_file + ".pt")
    print(model_file, " saved.")
    return model_file


def _parse_args():
    """Parse cmd line args for training an audio classifier."""
    parser = argparse.ArgumentParser(description="Train an audio classifier")
    parser = add_default_parser_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main()
