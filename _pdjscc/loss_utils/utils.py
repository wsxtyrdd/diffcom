import numpy as np
import matplotlib.pyplot as plt
import math
import os, time, datetime
import torch
import torchvision
import random
import os
from torch.autograd import Variable
import logging
from PIL import Image
from .loss.distortion import MS_SSIM


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def clear(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


def count_params(model):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))


def BLN2BCHW(x, H, W):
    B, L, N = x.shape
    return x.reshape(B, H, W, N).permute(0, 3, 1, 2)


def BCHW2BLN(x):
    return x.flatten(2).permute(0, 2, 1)


def CalcuPSNR(img1, img2, normalize=False, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64'
    if normalize:
        img1 = (torch.clamp(img1, -1, 1).cpu().numpy() + 1) / 2 * 255
        img2 = (torch.clamp(img2, -1, 1).cpu().numpy() + 1) / 2 * 255
    else:
        img1 = torch.clamp(img1, 0, 1).cpu().numpy() * 255
        img2 = torch.clamp(img2, 0, 1).cpu().numpy() * 255

    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


# def CalcuPSNR(target, ref):
#     diff = ref - target
#     diff = diff.flatten('C')
#     rmse = math.sqrt(np.mean(diff**2.))
#     return 20 * math.log10(1.0 / (rmse))


def MSE2PSNR(MSE):
    return 10 * math.log10(255 ** 2 / (MSE))


def logger_configuration(config, save_log=False, test_mode=False):
    # 配置 logger
    logger = logging.getLogger("Deep joint source channel coder")
    if test_mode:
        config.workdir += '_test'
    if save_log:
        makedirs(config.workdir)
        makedirs(config.samples)
        makedirs(config.models)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    if save_log:
        filehandler = logging.FileHandler(config.log)
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    config.logger = logger
    return config.logger


def Var(x, device):
    return Variable(x.to(device))


def single_plot(epoch, global_step, real, gen, config, normalize=True, single_compress=False):
    # print(real.shape)
    # real = real.permute(1, 2, 0)
    # gen = gen.permute(1, 2, 0)
    images = [real, gen]

    # comparison = np.hstack(images)

    # old save_fig

    # f = plt.figure()
    # plt.imshow(comparison)
    # plt.axis('off')
    # if single_compress:
    #     f.savefig(config.name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    # else:
    #     f.savefig("{}/JSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step),
    #               format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    # plt.gcf().clear()
    # plt.close(f)

    # new save_fig
    filename = "{}/JSCCModel_{}_epoch{}_step{}.png".format(config.samples, config.trainset, epoch, global_step)
    torchvision.utils.save_image(images, filename)


def triple_plot(filename, real, gen1, gen2):
    images = torch.cat((real, gen1, gen2), dim=0)
    torchvision.utils.save_image(images, filename)


def single_plot_II(epoch, global_step, real, gen_I, gen_II, config, single_compress=False):
    real = real.transpose([1, 2, 0])
    gen_I = gen_I.transpose([1, 2, 0])
    gen_II = gen_II.transpose([1, 2, 0])
    images = list()

    for im, imtype in zip([real, gen_I, gen_II], ['real', 'gen_I', 'gen_II']):
        im = ((im + 1.0)) / 2  # [-1,1] -> [0,1]
        im = np.squeeze(im)
        if len(im.shape) == 3:
            im = im[:, :, :3]
        if len(im.shape) == 4:
            im = im[0, :, :, :3]
        images.append(im)

    comparison = np.hstack(images)
    f = plt.figure()
    plt.imshow(comparison)
    plt.axis('off')
    if single_compress:
        f.savefig(config.name, format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    else:
        f.savefig("{}/gan_compression_{}_epoch{}_step{}_{}_comparison.png".format(config.samples, config.name, epoch,
                                                                                  global_step, imtype),
                  format='png', dpi=720, bbox_inches='tight', pad_inches=0)
    plt.gcf().clear()
    plt.close(f)


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter') != -1 and f.find('.model') != -1:
        st = f.find('iter') + 4
        ed = f.find('.model', st)
        return int(f[st:ed])
    else:
        return 0


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, base_lr, global_step, warmup_step, decay_interval, lr_decay):
    global cur_lr
    if global_step < warmup_step:
        lr = base_lr * global_step / warmup_step
    elif global_step < decay_interval:
        lr = base_lr
    else:
        lr = base_lr * lr_decay
    cur_lr = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Metrics:
    def __init__(self, logger, total_batches):
        self.metrics = dict({'mse': 0, 'perceptual': 0, 'disc': 0, 'gen': 0, 'bpp': 0, 'compression': 0})
        self.logger = logger

    def update(self, losses):
        for k, v in losses.items():
            self.metrics[k] += v

    def clear(self):
        self.metrics = dict({'mse': 0, 'perceptual': 0, 'disc': 0, 'gen': 0, 'bpp': 0, 'compression': 0})

    def log_metrics(self, epoch, batch_idx, total_batches, cur_lr, deltatime, bat_cnt):
        self.logger.info("【Training】 Train Epoch : {:02} [{:4}/{:4} ({:3.0f}%)] lr:{} time:{} ".format
                         (epoch, batch_idx, total_batches, 100. * batch_idx / total_batches, cur_lr,
                          (deltatime.seconds + 1e-6 * deltatime.microseconds) / bat_cnt))
        self.logger.info(
            "【Training】 Total : {:10.4f} G_loss : {:10.4f} D_loss : {:10.4f} bpp_loss : {:10.4f} mse_loss : "
            "{:10.4f} perceptual_loss : {:10.4f}".format
            (self.metrics['compression'] / bat_cnt, self.metrics['gen'] / bat_cnt, self.metrics['disc'] / bat_cnt,
             self.metrics['bpp'] / bat_cnt, self.metrics['mse'] / bat_cnt, self.metrics['perceptual'] / bat_cnt))


def testKodak(net, test_loader, logger, tb_logger, step):
    ms_ssim = MS_SSIM()
    with torch.no_grad():
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        cnt = 0
        for batch_idx, input in enumerate(test_loader):
            input = input.cuda()
            clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp = net(input)
            mse_loss, bpp_feature, bpp_z, bpp = \
                torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            sumBpp += bpp
            sumPsnr += psnr
            msssim = 1 - ms_ssim(clipped_recon_image, input)
            msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            logger.info(
                "Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(bpp, psnr, msssim, msssimDB))
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))
        sumBpp /= cnt
        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        logger.info("Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp,
                                                                                                                 sumPsnr,
                                                                                                                 sumMsssim,
                                                                                                                 sumMsssimDB))
        if tb_logger != None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("BPP_Test", sumBpp, step)
            tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
        else:
            logger.info("No need to add tensorboard")


def read_frame_to_torch(path):
    input_image = Image.open(path).convert('RGB')
    input_image = np.asarray(input_image).astype('float64').transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).type(torch.FloatTensor)
    input_image = input_image.unsqueeze(0) / 255
    return input_image


def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0) * 255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype('uint8'), 'RGB')
    frame_result.save(path)

def bpp_snr_to_kdivn(bpp, SNR):
    snr = 10 ** (SNR / 10)
    kdivn = bpp / 3 / np.log2(1 + snr)
    return kdivn

