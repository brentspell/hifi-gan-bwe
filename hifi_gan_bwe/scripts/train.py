""" HiFi-GAN+ model training script

This script trains a HiFi-GAN+ audio bandwidth extension model on the
VCTK dataset with noise augmentation from the DNS-Challenge dataset. The
HiFi-GAN+ model is trained in the GAN framework, where the generator is
first trained using only content losses (warmup phase), and then the
generator and discriminator are trained together during the joint phase.

Each training run is saved to a directory under the logs directory. All
runs have names that are suffixed with the current git hash
(ex: bwe-09-a81e232), which is used for simple experiment tracking,
reproducibility, and model provenance. These training runs contain model
checkpoints and the local state used to synchronize with the wandb.ai
experiment tracking site.

"""

import argparse
import typing as T
from pathlib import Path

import git
import numpy as np
import torch
import torchaudio
from matplotlib import pyplot as plt
from tqdm import tqdm

from hifi_gan_bwe import criteria, datasets, metrics, models

SAMPLE_RATE = datasets.SAMPLE_RATE
WARMUP_ITERATIONS = 100000
JOINT_ITERATIONS = 100000


class Trainer(torch.nn.Module):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        # load training, validation, and noise datasets
        self.train_set = datasets.VCTKDataset(args.vctk_path, training=True)
        self.valid_set = datasets.VCTKDataset(args.vctk_path, training=False)
        noise_set = datasets.DNSDataset(args.noise_path)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            collate_fn=datasets.Preprocessor(noise_set=noise_set, training=True),
            batch_size=datasets.BATCH_SIZE,
            shuffle=True,
            drop_last=True,
        )
        self.valid_loader = torch.utils.data.DataLoader(
            self.valid_set,
            collate_fn=datasets.Preprocessor(noise_set=noise_set, training=False),
            batch_size=datasets.BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )

        # create the generator and discriminator models
        self.gen_model = models.BandwidthExtender()
        self.dsc_model = models.Discriminator()
        self.gen_model.apply_weightnorm()

        # create loss criteria
        self.content_criteria = criteria.ContentCriteria()
        self.gan_criteria = torch.nn.MSELoss()
        self.feat_criteria = torch.nn.L1Loss()

        # create and configure the generator/discriminator optimizers
        self.gen_optimizer = torch.optim.Adam(self.gen_model.parameters(), lr=0.001)
        self.dsc_optimizer = torch.optim.Adam(self.dsc_model.parameters(), lr=0.001)
        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.gen_optimizer,
            lambda i: 1 if i < WARMUP_ITERATIONS else 0.01,
        )

        # create the log directory for this run, suffixed with the git hash
        git_hash = git.Repo().head.object.hexsha[:7]
        self.name = f"bwe-{args.name}-{git_hash}"
        self.log_path = args.log_path / self.name
        self.log_path.mkdir(parents=True, exist_ok=True)

        # create the mel spectrogram transformation for logging images
        self.melspec_xform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            f_min=8000,
            n_fft=2048,
            win_length=int(0.025 * SAMPLE_RATE),
            hop_length=int(0.010 * SAMPLE_RATE),
            n_mels=128,
            power=1,
        )

        # create the metrics summary, which is used for wandb logging
        self.metrics = metrics.Summary(
            project="hifi-gan-bwe",
            name=self.name,
            log_path=self.log_path,
            scalars=[
                metrics.Ema("gen_loss"),
                metrics.Ema("cnt_loss"),
                metrics.Ema("adv_loss"),
                metrics.Ema("feat_loss"),
                metrics.Ema("gen_grad"),
                metrics.Ema("gen_norm"),
                metrics.Mean("gen_fit"),
                metrics.Ema("dsc_loss"),
                metrics.Ema("dsc_grad"),
                metrics.Ema("dsc_norm"),
                metrics.Mean("dsc_fit"),
            ],
            use_wandb=not args.no_wandb,
        )

        self.iterations = 0

    def forward(
        self, batch: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> T.Dict:
        # determine whether to train the generator, discriminator, or both
        # . during the warmup phase, we only train the generator on content losses
        # . during joint training, we train both models, but we only train the
        #   generator every other iteration (as described in the paper)
        train_gen = self.iterations < WARMUP_ITERATIONS or self.iterations % 2 == 0
        train_dsc = self.iterations >= WARMUP_ITERATIONS

        gen_loss = None
        cnt_loss = None
        adv_loss = None
        feat_loss = None
        gen_grad = None
        gen_norm = None
        gen_fit = None
        dsc_loss = None
        dsc_grad = None
        dsc_norm = None
        dsc_fit = None

        x, r, y = batch

        # run the generator forward
        self.gen_optimizer.zero_grad()
        y_gen = self.gen_model(x, r)

        # -------------------------------------------------------------------
        # generator training
        # -------------------------------------------------------------------
        if not self.training or train_gen:
            # compute content losses
            gen_loss = cnt_loss = self.content_criteria(y_gen, y)

            # if we are in joint training compute adversarial and
            # feature map losses
            if train_dsc:
                y_dsc, f_dsc = self.dsc_model(torch.cat([y_gen, y], dim=0))
                y_fake, _y_real = y_dsc.chunk(2, dim=0)
                adv_loss = self.gan_criteria(y_fake, torch.ones_like(y_fake))
                feat_loss = sum(
                    (self.feat_criteria(*f.chunk(2, dim=0)) for f in f_dsc),
                    start=torch.tensor(0),
                ) / len(f_dsc)
                gen_loss += adv_loss + feat_loss

            # optimize the generator
            if self.training:
                gen_loss.backward()
                self.gen_optimizer.step()
                gen_grad = metrics.grad_norm(self.gen_model)
                gen_norm = metrics.weight_norm(self.gen_model)
            else:
                # if validating, just compute the overfit/underfit metric
                gen_fit = gen_loss / (self.metrics.scalars["gen_loss"] + 1e-5)

        # -------------------------------------------------------------------
        # discriminator training
        # -------------------------------------------------------------------
        if train_dsc:
            # run the discriminator forward
            self.dsc_optimizer.zero_grad()
            y_dsc, _f_dsc = self.dsc_model(torch.cat([y_gen.detach(), y], dim=0))
            y_fake, y_real = y_dsc.chunk(2, dim=0)

            # compute the discriminator loss
            y_true = torch.cat(
                [torch.zeros_like(y_fake), torch.ones_like(y_real)],
                dim=0,
            )
            dsc_loss = self.gan_criteria(y_dsc, y_true)

            # optimize the discriminator
            if self.training:
                dsc_loss.backward()
                self.dsc_optimizer.step()
                dsc_grad = metrics.grad_norm(self.dsc_model)
                dsc_norm = metrics.weight_norm(self.dsc_model)
            else:
                # if validating, just compute the overfit/underfit metric
                dsc_fit = dsc_loss / (self.metrics.scalars["dsc_loss"] + 1e-5)

        if self.training:
            self.iterations += 1
            self.gen_scheduler.step()

        # update and return the training/validation metrics
        results = dict(
            gen_loss=gen_loss,
            dsc_loss=dsc_loss,
            cnt_loss=cnt_loss,
            adv_loss=adv_loss,
            feat_loss=feat_loss,
            gen_grad=gen_grad,
            gen_norm=gen_norm,
            dsc_grad=dsc_grad,
            dsc_norm=dsc_norm,
        )
        results = {k: float(v) for k, v in results.items() if v is not None}
        if self.training:
            self.metrics.update(results)
        else:
            self.metrics.update(gen_fit=gen_fit, dsc_fit=dsc_fit)
        return results

    def evaluate(self) -> None:
        yt = [torch.from_numpy(y).cuda() for y in self.valid_set.eval_set]

        # create downsampled baselines for inference and comparison
        yt_8 = [torchaudio.functional.resample(y, SAMPLE_RATE, 8000) for y in yt]
        yt_16 = [torchaudio.functional.resample(y, SAMPLE_RATE, 16000) for y in yt]
        yt_24 = [torchaudio.functional.resample(y, SAMPLE_RATE, 24000) for y in yt]

        # create upsampled predictions from the bandwidth extender model
        with torch.no_grad():
            yp_8 = [self.gen_model(y, 8000) for y in yt_8]
            yp_16 = [self.gen_model(y, 16000) for y in yt_16]
            yp_24 = [self.gen_model(y, 24000) for y in yt_24]

        # send the audio samples to wandb for listening
        audios = dict(
            audio_true=(yt, SAMPLE_RATE),
            audio_true_8kHz=(yt_8, 8000),
            audio_true_16kHz=(yt_16, 16000),
            audio_true_24kHz=(yt_24, 24000),
            audio_pred_8kHz=(yp_8, SAMPLE_RATE),
            audio_pred_16kHz=(yp_16, SAMPLE_RATE),
            audio_pred_24kHz=(yp_24, SAMPLE_RATE),
        )
        for name, (y, sample_rate) in audios.items():
            self.metrics.audio(
                iterations=self.iterations,
                audio=torch.cat(y).cpu().numpy(),
                sample_rate=sample_rate,
                name=name,
            )

        # plot spectrograms for the baseline and predicted samples
        # send the spectrograms to wandb for rendering
        m_true = [torch.log(self.melspec_xform(y) + 1e-5) for y in yt]
        vmin = min(m.min() for m in m_true)
        vmax = max(m.max() for m in m_true)
        melspecs = dict(
            melspec_8kHz=[torch.log(self.melspec_xform(y) + 1e-5) for y in yp_8],
            melspec_16kHz=[torch.log(self.melspec_xform(y) + 1e-5) for y in yp_16],
            melspec_24kHz=[torch.log(self.melspec_xform(y) + 1e-5) for y in yp_24],
        )
        for name, ms in melspecs.items():
            fig, ax = plt.subplots(len(m_true), 2, figsize=(30, 5 * len(m_true)))
            ax[0][0].set_title("true")
            ax[0][1].set_title("pred")
            for ax, mt, mp in zip(ax, m_true, ms):
                for ax, m in zip(ax, [mt, mp]):
                    ax.imshow(
                        m.cpu(),
                        aspect="auto",
                        origin="lower",
                        vmin=vmin,
                        vmax=vmax,
                    )
            self.metrics.figure(
                iterations=self.iterations,
                figure=fig,
                name=name,
            )

    def load(self, checkpoint: str = "") -> None:
        ckpt_paths = sorted(self.log_path.glob(f"ckpt-{checkpoint or '*'}k.pt"))
        if checkpoint and not ckpt_paths:
            raise Exception(f"checkpoint {checkpoint} not found")
        if ckpt_paths:
            state = torch.load(ckpt_paths[-1])
            self.gen_model.load_state_dict(state["gen_model"])
            self.dsc_model.load_state_dict(state["dsc_model"])
            self.gen_optimizer.load_state_dict(state["gen_optimizer"])
            self.dsc_optimizer.load_state_dict(state["dsc_optimizer"])
            self.gen_scheduler.load_state_dict(state["gen_scheduler"])
            self.iterations = state["iterations"]

    def save(self) -> None:
        torch.save(
            dict(
                iterations=self.iterations,
                gen_model=self.gen_model.state_dict(),
                dsc_model=self.dsc_model.state_dict(),
                gen_optimizer=self.gen_optimizer.state_dict(),
                dsc_optimizer=self.dsc_optimizer.state_dict(),
                gen_scheduler=self.gen_scheduler.state_dict(),
            ),
            self.log_path / f"ckpt-{self.iterations // 10000 * 10:04d}k.pt",
        )
        self.metrics.save(self.iterations)


def main() -> None:
    parser = argparse.ArgumentParser("HiFi-GAN+ Bandwidth Extension Trainer")
    parser.add_argument(
        "name",
        help="training run name",
    )
    parser.add_argument(
        "--vctk_path",
        type=Path,
        default="./data/vctk",
        help="path to the VCTK speech dataset",
    )
    parser.add_argument(
        "--noise_path",
        type=Path,
        default="./data/dns",
        help="path to the DNS noise dataset",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default="./logs",
        help="training log root path",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="pass to disable Weights and Biases (wandb.ai) logging",
    )
    args = parser.parse_args()

    if git.Repo().is_dirty():
        print("warning: local git repo is dirty")

    # create the model trainer and load the latest checkpoint
    trainer = Trainer(args).cuda()
    trainer.load()

    # smoke test model evaluation before training
    trainer.eval()
    trainer.evaluate()
    trainer.train()

    print(trainer.gen_model)
    print(f"Params: {sum(np.prod(v.shape) for v in trainer.gen_model.parameters())}")

    # train the models
    results = {}
    total_iterations = WARMUP_ITERATIONS + JOINT_ITERATIONS
    with tqdm(initial=trainer.iterations, total=total_iterations) as pbar:
        pbar.set_description(trainer.name)
        while trainer.iterations < total_iterations:
            for batch in trainer.train_loader:
                # run a training step
                results.update(trainer(batch))
                pbar.update(1)
                pbar.set_postfix(**results)

                # periodically evaluate the model and save a checkpoint
                if trainer.iterations % 10000 == 0:
                    trainer.eval()
                    for batch in (pbar_eval := tqdm(trainer.valid_loader, leave=False)):
                        results = trainer(batch)
                        pbar_eval.set_postfix(**results)
                    trainer.evaluate()
                    trainer.save()
                    trainer.train()

                # periodically flush metrics to wandb
                if trainer.iterations % 100 == 0:
                    trainer.metrics.save(trainer.iterations)


if __name__ == "__main__":
    main()
