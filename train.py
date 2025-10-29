import argparse
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(logging_interval='epoch')

import cv2 as cv
import os
import torch.nn
import numpy as np
from torch.utils.data import DataLoader
import time
import warnings

from modules.utils.params import *
from modules.dataset.custom_dataset import Mycustomdataset
from modules.model.architecture8 import LitAutoEncoder
from modules.model.model_student import LitAutoEncoder_student
from modules.utils.save_web import save_web_format
from modules.dataset.datasetwmask import MLIC

warnings.filterwarnings("ignore")


def main():
    # -----------------------------
    # Argument Parser Configuration
    # -----------------------------
    parser = argparse.ArgumentParser(description="Train Teacher & Student AutoEncoder Models")

    parser.add_argument("--data_path", type=str, default="dataset/data",
                        help="Path to the input data folder.")
    parser.add_argument("--mask", action="store_true",
                        help="Use mask during training (default: False).", default=False)
    parser.add_argument("--src_img_type", type=str, default="jpg",
                        help="Source image type (e.g. png, jpg).")
    parser.add_argument("--output_path", type=str, default="outputs",
                        help="Directory to save Teacher and Student models.")

    args = parser.parse_args()

    # -----------------------------
    # Core script logic
    # -----------------------------
    t1 = time.time()

    teacher_spath = os.path.join(args.output_path, 'Teacher')
    student_spath = os.path.join(args.output_path, 'Student')

    os.makedirs(teacher_spath, mode=0o777, exist_ok=True)
    os.makedirs(student_spath, mode=0o777, exist_ok=True)

    # Initialize dataset
    mlic = MLIC(data_path=args.data_path, src_img_type=args.src_img_type, mask=args.mask)

    h, w = mlic.h, mlic.w
    unmasked_features = np.zeros((h * w, 9))

    hw, num_samples, no_channels = mlic.samples.shape
    sample_size = int(hw * num_samples)
    limit = hw
    samples = np.reshape(mlic.samples, (limit, -1))

    input_samples = torch.from_numpy(samples)
    np.random.seed(seed=42)
    all_indices = np.random.choice(sample_size, sample_size, replace=False)

    p_idx = all_indices % (hw)
    gt_idx = (all_indices * num_samples // sample_size).astype(np.uint8)
    pixel_index = np.zeros((sample_size, 2), dtype=int)
    pixel_index[:, 0] = p_idx
    pixel_index[:, 1] = gt_idx

    # Dataset for dataloader
    custom_dataset = Mycustomdataset(csv_file=pixel_index,
                                     info=[hw, num_samples, no_channels],
                                     data=[mlic.samples, mlic.ld])

    train_set, valid_set = torch.utils.data.random_split(custom_dataset, [0.9, 0.1])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=False)

    if args.mask:
        mask_img = mlic.binary_mask.flatten()
        masked_indices = np.squeeze(np.column_stack(np.where(mask_img == 255)))

    del mlic, samples, custom_dataset

    # -----------------------------
    # Train Teacher Network
    # -----------------------------
    trainer = L.Trainer(enable_model_summary=False, max_epochs=max_epochs,
                        check_val_every_n_epoch=1,
                        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5)])
    model = LitAutoEncoder(num_inputs=num_samples * no_channels)
    print("Training Teacher network")
    trainer.fit(model, train_loader, valid_loader)
    encoder = model.encoder
    decoder = model.decoder

    decoder_fpath = os.path.join(teacher_spath, "decoder.pth")
    encoder_fpath = os.path.join(teacher_spath, "encoder.pth")
    coeff_fpath = os.path.join(teacher_spath, "coefficient.npy")

    torch.save(decoder, decoder_fpath)
    torch.save(encoder, encoder_fpath)
    encoder.eval()

    with torch.no_grad():
        reconst_imgs = encoder(input_samples)
    features = reconst_imgs.cpu().numpy()
    np.save(coeff_fpath, features)

    max_f = [float(np.max(features[:, i])) for i in range(comp_coeff)]
    min_f = [float(np.min(features[:, i])) for i in range(comp_coeff)]
    bit_feat = 8
    for i in range(comp_coeff):
        features[:, i] = np.interp(features[:, i], (min_f[i], max_f[i]), (0, 2 ** bit_feat - 1))

    if args.mask:
        unmasked_features[masked_indices] = features
    else:
        unmasked_features = features

    features = np.reshape(unmasked_features, (h, w, comp_coeff))

    for j in range(comp_coeff // 3):
        cv.imwrite(os.path.join(teacher_spath, f'plane_{j}.jpg'),
                   features[..., 3 * j:3 * (j + 1)].astype(np.uint8))
        cv.imwrite(os.path.join(teacher_spath, f'plane_{j}.png'),
                   features[..., 3 * j:3 * (j + 1)].astype(np.uint8))

    save_web_format(decoder_fpath, coeff_fpath, h, w, comp_coeff, teacher_spath, num_samples)

    # -----------------------------
    # Train Student Network
    # -----------------------------
    print("Training student network")
    trainer_distillation = L.Trainer(enable_model_summary=False, max_epochs=max_epochs,
                                     check_val_every_n_epoch=1,
                                     callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=5),
                                                lr_monitor])
    model_distillation = LitAutoEncoder_student(num_inputs=num_samples * no_channels,
                                                teacher_model=teacher_spath,
                                                temperature=1, alpha=alpha)
    trainer_distillation.fit(model_distillation, train_loader, valid_loader)

    student_decoder_fpath = os.path.join(student_spath, "decoder.pth")
    student_encoder_fpath = os.path.join(student_spath, "encoder.pth")
    student_coeff_fpath = os.path.join(student_spath, "coefficient.npy")

    encoder_student = model_distillation.encoder
    decoder_student = model_distillation.decoder
    torch.save(decoder_student, student_decoder_fpath)

    with torch.no_grad():
        reconst_imgs = encoder_student(input_samples)
    features = reconst_imgs.cpu().numpy()
    np.save(student_coeff_fpath, features)

    max_f = [float(np.max(features[:, i])) for i in range(comp_coeff)]
    min_f = [float(np.min(features[:, i])) for i in range(comp_coeff)]
    for i in range(comp_coeff):
        features[:, i] = np.interp(features[:, i], (min_f[i], max_f[i]), (0, 2 ** bit_feat - 1))

    if args.mask:
        unmasked_features[masked_indices] = features
    else:
        unmasked_features = features

    features = np.reshape(unmasked_features, (h, w, comp_coeff))

    for j in range(comp_coeff // 3):
        cv.imwrite(os.path.join(student_spath, f'plane_{j}.jpg'),
                   features[..., 3 * j:3 * (j + 1)].astype(np.uint8))
        cv.imwrite(os.path.join(student_spath, f'plane_{j}.png'),
                   features[..., 3 * j:3 * (j + 1)].astype(np.uint8))

    torch.save(encoder_student, student_encoder_fpath)
    save_web_format(student_decoder_fpath, student_coeff_fpath, h, w, comp_coeff, student_spath, num_samples)

    t2 = time.time()
    print('done!')
    print(f'--- {int(t2 - t1) // 60 // 60} h {int(t2 - t1) // 60 % 60} m {int(t2 - t1) % 60} s ---')


if __name__ == "__main__":
    main()
