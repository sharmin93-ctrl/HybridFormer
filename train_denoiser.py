import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys

# add dir
from matplotlib import pyplot as plt
from torch.optim import SGD
import measure
from dataset_denoise import get_training_data, get_validation_data

from train.compound_loss import CompoundLoss

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(sys.path)
print(dir_name)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime
from measure import compute_PSNR
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR

######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'denoising', opt.dataset, opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
now = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
logname = os.path.join(log_dir, now + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.99), eps=1e-8,
                            weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'nadm':
    optimizer = optim.NAdam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.99), eps=1e-8,
                            weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'sgd':
    optimizer = SGD(model_restoration.parameters(),
                    lr=opt.lr_initial,
                    momentum=0.9,
                    dampening=0,
                    weight_decay=opt.weight_decay,
                    nesterov=False)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()
######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()
######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print("Resume from " + path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

######### Loss ###########
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# criterion = MSELoss().cuda()
# criterion = CharbonnierLoss().cuda()
criterion = CompoundLoss()
# criterion = nn.L1Loss()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=False,
                          num_workers=opt.train_workers, pin_memory=True, drop_last=False)
val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=True, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset, ", sizeof validation set: ", len_valset)


def denormalize_(image):
    image = image * (3072.0 - -1024.0) + -1024.0
    return image


def trunc(mat):
    mat[mat <= -160] = -160
    mat[mat >= 240] = 240
    return mat


def save_fig(x, y, pred, fig_name, original_result, pred_result):
    x, y, pred = x.numpy(), y.numpy(), pred.numpy()  # 将Tensor变量转换为ndarray变量  Convert the Tensor variable to the ndarray variable
    f, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].imshow(x, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[0].set_title('Quarter-dose', fontsize=30)
    ax[0].set_xlabel("PSNR: {:.4f}".format(original_result), fontsize=20)
    ax[1].imshow(pred, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[1].set_title('Result', fontsize=30)
    ax[1].set_xlabel("PSNR: {:.4f}".format(pred_result), fontsize=20)
    ax[2].imshow(y, cmap=plt.cm.gray, vmin=-160, vmax=240)
    ax[2].set_title('Full-dose', fontsize=30)

    plt.close()

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
eval_now = len(train_loader)
print("\nEvaluation after every {} Iterations !!!\n".format(eval_now))
lr_list = []  # 学习率存储数组  The learning rate storage array
loss_list = []  # 损失存储数组  Loss storage array
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    model_restoration.train()
    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        optimizer.zero_grad()

        # target = data[0].float().cuda()
        # input_ = data[1].float().cuda()
        target = data[0].unsqueeze(0).float().cuda()
        input_ = data[1].unsqueeze(0).float().cuda()
        target = target.view(-1, 1, opt.train_ps, opt.train_ps)
        input_ = input_.view(-1, 1, opt.train_ps, opt.train_ps)
        # with torch.cuda.amp.autocast():
        restored = model_restoration(input_)
        loss = criterion(restored, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # clear last step
        optimizer.zero_grad()

        # Evaluation ####
        if (i + 1) % eval_now == 0 and i > 0 and epoch == 1:
            with torch.no_grad():
                model_restoration.eval()
                psnr_val_rgb = []
                for ii, (target, input_, targetImagePath, inputImagePath) in enumerate(tqdm(val_loader), 0):
                    shape_ = input_.shape[-1]
                    input_ = input_.float().cuda()
                    target = target.float().cuda()

                    # with torch.cuda.amp.autocast():
                    restored = model_restoration(input_)
                    input_ = trunc(denormalize_(input_.view(shape_, shape_).cpu().detach()))
                    target = trunc(denormalize_(target.view(shape_, shape_).cpu().detach()))
                    restored = trunc(denormalize_(restored.view(shape_, shape_).cpu().detach()))
                    pred_result = measure.compute_PSNR(restored, target, 400)
                    original_result = measure.compute_PSNR(input_, target, 400)
                    psnr_val_rgb.append(compute_PSNR(restored, target, 400))
                    save_fig(input_, target, restored, ii, original_result, pred_result)
                psnr_val_rgb = sum(psnr_val_rgb) / len_valset

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))
                    # torch.save(model_restoration.state_dict(), os.path.join(model_dir, "model_best.pth"))
                print(
                    "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
                with open(logname, 'a') as f:
                    f.write(
                        "[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr) + '\n')
                model_restoration.train()
                torch.cuda.empty_cache()
    loss_list.append(epoch_loss)  # 插入每次训练的损失值  Insert the loss value of each training
    lr_list.append(scheduler.get_lr())
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                                epoch_loss,
                                                                                scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))
    if epoch % opt.checkpoint == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))
print("Now time is : ", datetime.datetime.now().isoformat())

# 绘制训练时损失曲线
plt.figure(1)
x = range(0, opt.nepoch)
plt.xlabel('epoch')
plt.ylabel('epoch loss')
plt.plot(x, loss_list, 'r-')
# 绘制学习率改变曲线
plt.figure(2)
plt.xlabel('epoch')
plt.ylabel('learning rate')
plt.plot(x, lr_list, 'r-')

plt.show()
