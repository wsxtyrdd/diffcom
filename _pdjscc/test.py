import pyiqa

from loss_utils.utils import *
from config import config
from data.dataset import get_test_loader
from net.network import ADJSCC


def CalcuPSNR(img1, img2, max_val=255.):
    """
    Based on `tf.image.psnr`
    https://www.tensorflow.org/api_docs/python/tf/image/psnr
    """
    float_type = 'float64'
    # img1 = (torch.clamp(img1,-1,1).cpu().numpy() + 1) / 2 * 255
    # img2 = (torch.clamp(img2,-1,1).cpu().numpy() + 1) / 2 * 255
    img1 = torch.clamp(img1, 0, 1).cpu().numpy() * 255
    img2 = torch.clamp(img2, 0, 1).cpu().numpy() * 255
    img1 = img1.astype(float_type)
    img2 = img2.astype(float_type)
    mse = np.mean(np.square(img1 - img2), axis=(1, 2, 3))
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


config.device = torch.device("cuda:0")
logger = config.logger = logger_configuration(config, save_log=False, test_mode=True)

lpips_metric = pyiqa.create_metric('lpips').to(config.device)
dists_metric = pyiqa.create_metric('dists').to(config.device)
niqe_metric = pyiqa.create_metric('niqe').to(config.device)

# initialize model
config.C = 8
config.multiple_snr = [0]
config.test_data_dir = ["./demo"]
save_dir = './recon/demo/{}dB/C={}/recon'.format(config.multiple_snr[0], config.C)

net = ADJSCC(config)
net = net.cuda()
num_params = 0
for param in net.parameters():
    num_params += param.numel()
print("TOTAL Params {}M".format(num_params / 10 ** 6))

pre_dict = torch.load("./pretrained_model/CBR=0.08.model",
                      map_location=config.device)
net.load_state_dict(pre_dict, strict=False)

# initialize dataset
test_loader = get_test_loader(config)
from loss_utils.utils import *

avg_time = 0
net.eval()
global_step = -1
with torch.no_grad():
    elapsed, ms_ssim_losses, mse_losses, lpips_losses, psnrs, dists_losses, niqe = [AverageMeter() for _ in range(7)]
    metrics = [elapsed, mse_losses, ms_ssim_losses, lpips_losses, psnrs, dists_losses, niqe]
    for batch_idx, input_image in enumerate(test_loader):
        global_step += 1
        start_time = time.time()
        input_image = input_image.cuda()
        ms_ssim_loss, mse_loss, lpips_loss, x_hat = net.forward(input_image)
        niqe_loss = niqe_metric(x_hat)
        niqe.update(niqe_loss.item())
        elapsed.update(time.time() - start_time)
        ms_ssim_losses.update(ms_ssim_loss.item())
        mse_losses.update(mse_loss.item())
        lpips_losses.update(lpips_loss.item())
        dists_loss = dists_metric(input_image, x_hat)
        dists_losses.update(dists_loss.item())
        if mse_loss.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse_loss) / np.log(10))
            psnrs.update(psnr.item())
        else:
            psnrs.update(100)
        log = (' | '.join([
            f'test_Time {elapsed.avg:.2f}',
            f'test_NIQE {niqe.val:.3f} ({niqe.avg:.3f})',
            f'test_PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
            f'test_LPIPS {lpips_losses.val:.3f} ({lpips_losses.avg:.3f})',
            f'test_MS-SSIM {ms_ssim_losses.val:.3f} ({ms_ssim_losses.avg:.3f})',
            f'test_dists {dists_losses.val:.3f} ({dists_losses.avg:.3f})',
        ]))

        # makedirs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fname = os.path.join(save_dir,
                             "{}.png".format(global_step.__str__().zfill(5)))
        torchvision.utils.save_image(x_hat, fname, normalize=True)
        logger.info(log)
    for i in metrics:
        i.clear()
