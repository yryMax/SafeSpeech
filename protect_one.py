
import torch
import argparse

from torch.utils.data import DataLoader

import soundfile as sf

import sys
sys.path.append("bert_vits2/")
import bert_vits2.utils as utils
from bert_vits2.data_utils import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
)
from bert_vits2.losses import WavLMLoss
from toolbox import build_models_noise, build_optims
from protect import perturb
from tqdm import tqdm
from bert_vits2.text.cleaner import clean_text
from bert_gen import process_line
from bert_vits2.text import check_bert_models
from multiprocessing import Pool


weight_alpha = 0.05
weight_beta = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SafeSpeech")
    parser.add_argument('--filepath', type=str, default="filelists/libritts_train_text.txt",
                        help='the file list of fine-tuning dataset')
    parser.add_argument("--output_path", type=str, default="protected.wav",
                        help="Where to save the perturbed audio.")

    args = parser.parse_args()

    device = torch.device("cuda")
    model_name = "BERT_VITS2"
    dataset_name = "LibriTTS"
    mode = "SPEC"
    config_path = f"bert_vits2/configs/{dataset_name.lower()}_{model_name.lower()}.json"
    hps = utils.get_hparams_from_file(config_path)
    hps.train.batch_size = 27
    hps.model_dir = "checkpoints/base_models"


    # preprocess_text.py
    txt_path = args.filepath
    with open(txt_path, "r") as f:
        lines = f.readlines()

    tmp_output_path = "filelists/tmp.cleaned"
    with open(tmp_output_path, "w") as f:
        for line in tqdm(lines):
            utt, spk, text = line.strip().split("|")
            language = "EN"
            norm_text, phones, tones, word2ph = clean_text(
                text, language
            )
            f.write(
                "{}|{}|{}|{}|{}|{}|{}\n".format(
                    utt,
                    spk,
                    language,
                    norm_text,
                    " ".join(phones),
                    " ".join([str(i) for i in tones]),
                    " ".join([str(i) for i in word2ph]),
                )
            )


    # bert_gen.py

    check_bert_models()
    lines = []

    txt = f"filelists/tmp.cleaned"
    with open(txt, encoding="utf-8") as f:
        lines.extend(f.readlines())

    add_blank = [hps.data.add_blank] * len(lines)

    if len(lines) != 0:
        num_processes = 1
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                    pool.imap_unordered(process_line, zip(lines, add_blank)),
                    total=len(lines),
            ):
                pass
    # protect.py

    torch.manual_seed(hps.train.seed)
    torch.cuda.manual_seed(hps.train.seed)
    train_dataset = TextAudioSpeakerLoader(txt, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    train_loader = DataLoader(train_dataset,
                              num_workers=4,
                              shuffle=False,
                              collate_fn=collate_fn,
                              batch_size=hps.train.batch_size,
                              pin_memory=True,
                              drop_last=False)

    nets = build_models_noise(hps, device)
    net_g, net_d, net_wd, net_dur_disc = nets

    optims = build_optims(hps, nets)
    optim_g, optim_d, optim_wd, optim_dur_disc = optims

    dur_resume_lr = hps.train.learning_rate
    wd_resume_lr = hps.train.learning_rate

    _, _, dur_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "DUR_*.pth"),
        net_dur_disc,
        optim_dur_disc,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_dur_disc.param_groups[0].get("initial_lr"):
        optim_dur_disc.param_groups[0]["initial_lr"] = dur_resume_lr

    _, optim_g, g_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
        net_g,
        optim_g,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    _, optim_d, d_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
        net_d,
        optim_d,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_g.param_groups[0].get("initial_lr"):
        optim_g.param_groups[0]["initial_lr"] = g_resume_lr
    if not optim_d.param_groups[0].get("initial_lr"):
        optim_d.param_groups[0]["initial_lr"] = d_resume_lr

    epoch_str = max(epoch_str, 1)
    global_step = int(utils.get_steps(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")))

    _, optim_wd, wd_resume_lr, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "WD_*.pth"),
        net_wd,
        optim_wd,
        skip_optimizer=(
            hps.train.skip_optimizer if "skip_optimizer" in hps.train else True
        ),
    )
    if not optim_wd.param_groups[0].get("initial_lr"):
        optim_wd.param_groups[0]["initial_lr"] = wd_resume_lr

    wl = WavLMLoss(
        hps.model.slm.model,
        net_wd,
        hps.data.sampling_rate,
        hps.model.slm.sr,
    ).to(device)



    noises = [None] * len(train_loader)
    max_epoch = 200
    epsilon = 8 / 255
    alpha = epsilon / 10

    for param in net_g.parameters():
        param.requires_grad = False
    for param in net_d.parameters():
        param.requires_grad = False
    for param in net_dur_disc.parameters():
        param.requires_grad = False
    for param in net_wd.parameters():
        param.requires_grad = False
    for param in wl.parameters():
        param.requires_grad = False

    for batch_index, batch_data in enumerate(train_loader):
        loss, noise = perturb(hps, net_g, batch_data, epsilon, alpha, max_epoch,
                                            [weight_alpha, weight_beta], mode, device)

        wav_tensor = batch_data[4]  # [1, 1, T]
        sr = hps.data.sampling_rate

        protected_wav = (wav_tensor.cpu() + noise.cpu()).clamp(-1., 1.).detach().cpu().numpy().squeeze()
        sf.write(args.output_path, protected_wav, sr)
        torch.cuda.empty_cache()
        print(f"[INFO] Saved protected audio to {args.output_path}")
        break