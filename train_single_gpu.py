import os
import random
import torch
import numpy as np
from model.preprocess_rfdn import RFLite_v3
from model.loss_ssim import SSIMLoss
from model.loss_dct import LowRankLoss, LowRankLoss16
from datasets import TrainDataset, TrainDataset_real, TrainDataset_real_paired, ValDataset, ValDataset_real
from utils.utils_image import calculate_psnr, calculate_ssim, tensor2single, single2uint, img_save, filter2D
from utils.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from utils import diffjpeg
from torch.utils.tensorboard import SummaryWriter
try:
    # For newer PyTorch versions
    from torch.amp import GradScaler, autocast
except ImportError:
    # For older PyTorch versions
    from torch.cuda.amp import GradScaler, autocast


class SingleGPUTrainer:
    def __init__(self, config, degradation_config):
        self.config = config
        self.degradation_config = degradation_config
        self.rank = 0  # Single GPU, always rank 0
        self.world_size = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.init_datasets()
        self.train(self.rank, self.world_size)

    def init_datasets(self):

        if self.config['degradation_type'] == 'real':
            print(".......... real degradation ..............")
            self.train_set = TrainDataset_real(self.config, self.degradation_config)
            self.val_set = ValDataset_real(self.config, self.degradation_config)
        elif self.config['degradation_type'] == 'paired':
            print(".......... paired degradation ..............")
            self.train_set = TrainDataset_real_paired(self.config, self.degradation_config)
            self.val_set = ValDataset_real(self.config, self.degradation_config)
        else:
            print(".......... norm degradation ..............")
            self.train_set = TrainDataset(self.config, self.degradation_config)
            self.val_set = ValDataset(self.config)

        # Single GPU mode - no distributed sampler
        # Windows compatibility: reduce workers if needed
        import platform
        max_workers = self.config["num_workers"]
        if platform.system() == "Windows" and max_workers > 4:
            print(f"Windows detected: reducing num_workers from {max_workers} to 4 for stability")
            max_workers = 4

        self.train_loader = torch.utils.data.DataLoader(self.train_set,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=max_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True if max_workers > 0 else False)

        self.val_loader = torch.utils.data.DataLoader(self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=2,  # 验证用更少的worker
            drop_last=False,
            pin_memory=True
            )

    def output_log(self, log_str):
        log_file_path = os.path.join(self.config["output_dir"], "train_log.txt")
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        with open(log_file_path, "a") as f:
            f.write(log_str)

    def init_model(self):
        self.model = RFLite_v3()

    def init_loss_and_optimizer(self):

        self.l1_criterion = torch.nn.L1Loss().to(self.device)
        self.ssim_criterion = SSIMLoss().to(self.device)
        self.lowrank_criterion = LowRankLoss().to(self.device)
        self.lowrank16_criterion = LowRankLoss16().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config["scheduler_milestones"], gamma=self.config["scheduler_gamma"])

        # Mixed precision training
        self.use_amp = self.config.get("mixed_precision", True)
        if self.use_amp:
            try:
                # For newer PyTorch versions
                self.scaler = GradScaler('cuda')
            except TypeError:
                # For older PyTorch versions
                self.scaler = GradScaler()
            print("Mixed precision training enabled")

        self.jpeger = diffjpeg.DiffJPEG(differentiable=False).cuda()

    @torch.no_grad()
    def real_degradations(self, batch):

        opt = self.degradation_config
        L = batch["L"].to(self.device)  # low-quality image
        H = batch["H"].to(self.device)
        kernel = batch["kernel"].to(self.device)
        kernel2 = batch["kernel2"].to(self.device)
        sinc_kernel = batch["sinc_kernel"].to(self.device)

        # ----------------------- The first degradation process ----------------------- #
        # blur
        if np.random.uniform() < 0.9:
            out = filter2D(H, kernel)
        else:
            out = H

        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = torch.nn.functional.interpolate(
                out, size=(int(self.config["patch_size"]  * scale), int(self.config["patch_size"]  * scale)), mode=mode)
        # noise
        if opt['gaussian_noise_prob']>0:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=opt['gray_noise_prob'])
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range'],
                gray_prob=opt['gray_noise_prob'],
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            out = torch.nn.functional.interpolate(out, size=(self.config["patch_size"] , self.config["patch_size"] ), mode='bicubic')
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            out = torch.nn.functional.interpolate(out, size=(self.config["patch_size"] , self.config["patch_size"] ), mode='bicubic')
            out = filter2D(out, sinc_kernel)

        return out

    def train(self, rank, world_size):
        print("\n======================================================================")
        print("==================== Training Start (Single GPU) ====================")
        print("======================================================================")

        self.output_log("\n======================================================================\n")
        self.output_log("==================== Training Start (Single GPU) ====================\n")
        self.output_log("======================================================================\n")

        self.init_model()
        self.model.to(self.device)
        self.init_loss_and_optimizer()
        current_step = 0
        self.config["best_psnr"] = 0

        # Initialize TensorBoard
        tb_dir = os.path.join(self.config["output_dir"], "tensorboard")
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir, flush_secs=60)

        pretrained = self.config["network"]["pretrained"]
        if pretrained and os.path.exists(pretrained):
            print(f"Loading pretrained checkpoint from {pretrained}")
            self.output_log(f"Loading pretrained checkpoint from {pretrained}\n")
            checkpoint = torch.load(pretrained, map_location=self.device)
            self.model.load_state_dict(checkpoint['net_g_ema'], strict=False)
            current_step = checkpoint['iter'] if 'iter' in checkpoint else checkpoint.get('step', 0)
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])

        try:
            for epoch in range(1000):
                print(f"\nEpoch {epoch}: current_step = {current_step}")

                for _, batch in enumerate(self.train_loader):
                    current_step += 1
                    L = batch["L"].to(self.device)  # low-quality image
                    H = batch["H"].to(self.device)

                    if self.config['degradation_type'] == 'real' or self.config['degradation_type'] == 'paired':
                        L = self.real_degradations(batch)

                    # Training
                    self.model.train()
                    self.optimizer.zero_grad()

                    if self.use_amp:
                        try:
                            # For newer PyTorch versions
                            with autocast('cuda'):
                                output = self.model(L)
                                l1_loss = self.l1_criterion(output, H)
                                ssim_loss = self.ssim_criterion(output, H)
                                lowrank_loss = self.lowrank_criterion(output)
                                lowrank16_loss = self.lowrank16_criterion(output)
                                loss = l1_loss + ssim_loss * (-0.1) + lowrank_loss * 8 + lowrank16_loss * 8
                        except TypeError:
                            # For older PyTorch versions
                            with autocast():
                                output = self.model(L)
                                l1_loss = self.l1_criterion(output, H)
                                ssim_loss = self.ssim_criterion(output, H)
                                lowrank_loss = self.lowrank_criterion(output)
                                lowrank16_loss = self.lowrank16_criterion(output)
                                loss = l1_loss + ssim_loss * (-0.1) + lowrank_loss * 8 + lowrank16_loss * 8

                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        output = self.model(L)
                        l1_loss = self.l1_criterion(output, H)
                        ssim_loss = self.ssim_criterion(output, H)
                        lowrank_loss = self.lowrank_criterion(output)
                        lowrank16_loss = self.lowrank16_criterion(output)
                        loss = l1_loss + ssim_loss * (-0.1) + lowrank_loss * 8 + lowrank16_loss * 8

                        loss.backward()
                        self.optimizer.step()
                    self.scheduler.step()

                    # Logging
                    if current_step % 100 == 0:
                        lr = self.optimizer.param_groups[0]['lr']

                        # GPU memory stats
                        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                        gpu_memory_cached = torch.cuda.memory_reserved() / 1024**3  # GB

                        print(f"Step {current_step}: Loss = {loss:.4f}, L1 = {l1_loss:.4f}, "
                              f"SSIM = {ssim_loss:.4f}, LR8 = {lowrank_loss:.4f}, LR16 = {lowrank16_loss:.4f}, "
                              f"LR = {lr}, GPU Mem: {gpu_memory:.1f}GB/{gpu_memory_cached:.1f}GB")
                        self.output_log(f"Step {current_step}: Loss = {loss:.4f}, L1 = {l1_loss:.4f}, "
                                       f"SSIM = {ssim_loss:.4f}, LR8 = {lowrank_loss:.4f}, LR16 = {lowrank16_loss:.4f}, LR = {lr}\n")

                        # TensorBoard logging
                        self.writer.add_scalar("loss/train", loss.item(), current_step)
                        self.writer.add_scalar("L1_loss/train", l1_loss.item(), current_step)
                        self.writer.add_scalar("SSIM_loss/train", ssim_loss.item(), current_step)
                        self.writer.add_scalar("LowRankLoss/train", lowrank_loss.item(), current_step)
                        self.writer.add_scalar("LowRankLoss16/train", lowrank16_loss.item(), current_step)
                        self.writer.add_scalar("learning_rate", lr, current_step)
                        self.writer.add_scalar("GPU_memory_GB", gpu_memory, current_step)

                    # Validation and checkpoint saving
                    if current_step % self.config["steps_val"] == 0:
                        self.validate(current_step)
                        self.save_checkpoint(current_step)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            self.cleanup()

    def validate(self, current_step):
        print(f"\nValidating at step {current_step}...")
        self.model.eval()
        psnr_list, ssim_list = [], []

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                if idx >= 10:  # Validate on first 10 images
                    break

                L = batch["L"].to(self.device)
                H = batch["H"].to(self.device)

                if self.config['degradation_type'] == 'real' or self.config['degradation_type'] == 'paired':
                    L = self.real_degradations(batch)

                output = self.model(L)

                # Calculate metrics
                psnr = calculate_psnr(output * 255.0, H * 255.0, input_order='CHW')
                ssim = calculate_ssim(output * 255.0, H * 255.0, input_order='CHW')
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                # Save validation images
                if idx < 5:
                    self.save_validation_images(L, output, H, current_step, idx)

        avg_psnr = sum(psnr_list) / len(psnr_list)
        avg_ssim = sum(ssim_list) / len(ssim_list)
        print(f"Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        self.output_log(f"Step {current_step} - Validation PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}\n")

        # TensorBoard logging for validation metrics
        self.writer.add_scalar("PSNR/val", avg_psnr, current_step)
        self.writer.add_scalar("SSIM/val", avg_ssim, current_step)

        # Save best model
        if avg_psnr > self.config["best_psnr"]:
            print(f"New best PSNR: {avg_psnr:.2f}")
            self.save_best_checkpoint(current_step, avg_psnr)
            self.config["best_psnr"] = avg_psnr

    def save_validation_images(self, L, output, H, current_step, idx):
        val_dir = os.path.join(self.config["output_dir"], "validation", f"step_{current_step}")
        os.makedirs(val_dir, exist_ok=True)

        # Convert tensors to numpy arrays and then to uint8
        L_img = single2uint(tensor2single(L))
        output_img = single2uint(tensor2single(output))
        H_img = single2uint(tensor2single(H))

        # Create residual image
        residual = torch.abs(output - H)
        residual_img = single2uint(tensor2single(residual))

        # Concatenate images horizontally
        concat_img = np.concatenate([L_img, output_img, H_img, residual_img], axis=1)

        # Save concatenated image
        img_save(concat_img, os.path.join(val_dir, f"val_{idx:02d}.png"))

    def save_checkpoint(self, current_step):
        checkpoint = {
            'step': current_step,
            'net_g_ema': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_path = os.path.join(self.config["output_dir"], "checkpoints", f"step_{current_step}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at {save_path}")

    def save_best_checkpoint(self, current_step, psnr):
        checkpoint = {
            'step': current_step,
            'psnr': psnr,
            'net_g_ema': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        save_path = os.path.join(self.config["output_dir"], "checkpoints", "model_best_psnr.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Best model checkpoint saved at {save_path}")

    def cleanup(self):
        # Close TensorBoard writer
        if hasattr(self, 'writer'):
            self.writer.close()
            print("TensorBoard writer closed.")