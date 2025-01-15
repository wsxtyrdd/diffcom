import sys
from config import config
from data.dataset import get_loader
from net.network import ADJSCC
import torch.optim as optim
from utils import *
import time
from loss_utils.utils import *



# initialize model
config.batch_size = 8
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_list
logger = logger_configuration(config, save_log=True)
logger.info(config.__dict__)
net = ADJSCC(config)
if len(config.gpu_list.split(',')) > 1:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()
logger.info(net)
predict = torch.load(config.predict)
net.load_state_dict(predict, strict=False)
net = net.cuda()
G_params = set(p for n, p in net.named_parameters() if not n.startswith("Discriminator"))

optimizer_G = optim.Adam(G_params, lr=config.g_learning_rate)
if config.use_discriminator:
    D_params = set(p for n, p in net.named_parameters() if n.startswith("Discriminator"))
    optimizer_D = optim.Adam(D_params, lr=config.d_learning_rate)

global_step = 0
train_generator = False
train_discriminator_steps = 0
train_generator_steps = 0

# load dataset
trainloader, testloader = get_loader(config)
global_step = 0

# model training
for epoch in range(config.epochs):
    net.train()
    device = next(net.parameters()).device
    elapsed, losses,ms_ssim_losses,mse_losses,lpips_losses,d_loss, g_loss,d_gen,g_real,d_real,psnrs = [AverageMeter() for _ in range(11)]
    metrics = [elapsed, losses,mse_losses,ms_ssim_losses,lpips_losses,d_loss, g_loss,d_gen,g_real,d_real,psnrs]
    for batch_idx, input_image in enumerate(trainloader):
        start_time = time.time()
        input_image = input_image.to(device)
        global_step += 1
        if config.use_discriminator:
            ms_ssim_loss,mse_loss,lpips_loss, x_hat,D_loss, G_loss,D_gen,D_real = net.forward(input_image,train_generator)
            if train_generator == True and D_gen < config.dis_acc:
                loss = config.K_M*mse_loss+config.K_P*lpips_loss +config.beta*G_loss
                optimizer_G.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer_G.step()
                train_generator_steps += 1
                if train_generator_steps  == config.generator_steps :
                    train_generator = False
                    train_generator_steps = 0
            else:
                loss = D_loss
                optimizer_D.zero_grad()
                loss.backward()
                # for name, parms in net.named_parameters():
                #     if name.startswith("Discriminator"):
                #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #               ' -->grad_value:', parms.grad)
                optimizer_D.step()
                train_discriminator_steps +=1
                if train_discriminator_steps  == config.discriminator_steps :
                    train_generator = True
                    train_discriminator_steps = 0
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            mse_losses.update(mse_loss.item())
            ms_ssim_losses.update(ms_ssim_loss.item())
            lpips_losses.update(lpips_loss.item())
            d_gen.update(D_gen)
            d_real.update(D_real)
            g_loss.update(G_loss.item())
            d_loss.update(D_loss.item())
        else:
            ms_ssim_loss, mse_loss, lpips_loss, x_hat = net.forward(input_image)
            # print('recon_Image:{}'.format(x_hat))
            # print(mse_loss)
            loss = config.K_M * mse_loss + config.K_P * lpips_loss
            optimizer_G.zero_grad()
            loss.backward()
            # for name, parms in net.named_parameters():
            #     if name.startswith("Decoder.conv_block_out"):
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #               ' -->grad_value:', parms.grad)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer_G.step()
            elapsed.update(time.time() - start_time)
            losses.update(loss.item())
            mse_losses.update(mse_loss.item())
            ms_ssim_losses.update(ms_ssim_loss.item())
            lpips_losses.update(lpips_loss.item())

        if mse_loss.item() > 0:
            psnr = 10 * (torch.log(255. * 255. / mse_loss) / np.log(10))
            psnrs.update(psnr.item())
        else:
            psnrs.update(100)

        if (global_step % config.print_step) == 0:
            process = (global_step % trainloader.__len__()) / (trainloader.__len__()) * 100.0
            log = (' | '.join([
                f'Step [{global_step % trainloader.__len__()}/{trainloader.__len__()}={process:.2f}%]',
                f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                f'Time {elapsed.avg:.2f}',
                f'PSNR {psnrs.val:.2f} ({psnrs.avg:.2f})',
                f'LPIPS {lpips_losses.val:.3f} ({lpips_losses.avg:.2f})',
                f'MS-SSIM {ms_ssim_losses.val:.3f} ({ms_ssim_losses.avg:.3f})',
                f'G_loss {g_loss.val:.3f}({g_loss.avg:.3f})',
                f'D_loss {d_loss.val:.3f}({d_loss.avg:.3f})',
                f'D_gen {d_gen.val:.3f} ({d_gen.avg:.3f})',
                f'D_real {d_real.val:.3f} ({d_real.avg:.3f})',
                f'Epoch {epoch}',
                f'Lr {config.g_learning_rate}',
            ]))
            logger.info(log)
            for i in metrics:
                i.clear()
        if (global_step % config.test_step) == 0:
            net.eval()
            with torch.no_grad():
                for batch_idx, input_image in enumerate(testloader):
                    start_time = time.time()
                    input_image = input_image.cuda()
                    ms_ssim_loss,mse_loss, lpips_loss, x_hat = net.forward(input_image)
                    elapsed.update(time.time() - start_time)
                    losses.update(loss.item())
                    ms_ssim_losses.update(ms_ssim_loss.item())
                    mse_losses.update(mse_loss.item())
                    lpips_losses.update(lpips_loss.item())
                    if mse_loss.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse_loss) / np.log(10))
                        psnrs.update(psnr.item())
                    else:
                        psnrs.update(100)
                    log = (' | '.join([
                        f'test_Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'test_Time {elapsed.avg:.2f}',
                        f'test_PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'test_LPIPS {lpips_losses.val:.3f} ({lpips_losses.avg:.3f})',
                        f'test_MS-SSIM {ms_ssim_losses.val:.3f} ({ms_ssim_losses.avg:.3f})'
                    ]))
                    logger.info(log)
                for i in metrics:
                    i.clear()
                net.train()

        if global_step % 5000 == 0 and global_step > 1:
            save_model(net,
                       save_path=config.models + '/{}_EP{}_Step{}.model'.format(config.filename, epoch, global_step))
