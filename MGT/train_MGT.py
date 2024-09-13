
import torch
import warnings
import os.path
from torch.utils.data import DataLoader
from tqdm import tqdm
from MGT import MGT
from data_5_fiter import TrainDataset
from loss import SwinFusion_loss_5 as Swin_loss

warnings.filterwarnings("ignore")
EPSILON = 1e-5


def main():
    # GPU/CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # method type and path
    method_type = 'MGT'
    model_load = False
    model_path = './Model/' + f'{method_type}/{method_type}_epoch_x.pth'
    optimizer_path = './Model/' + f'{method_type}/optimizer_epoch_x.pth'
    # Dataset
    Vis_RGB = False
    trainset_type = "MSRS"  # MSRS/TNO
    if trainset_type == 'MSRS':
        Vis_RGB = True
    batch_size = 4
    # model save path
    model_save_path = os.path.join('./Model/', method_type)
    if os.path.exists(model_save_path) is not True:
        os.makedirs(model_save_path)

    # load model
    model = MGT(Ex_depths=3, Fusion_depths=3, Re_depths=3)
    model.to(device)
    if model_load:
        model.load_state_dict(torch.load(model_path))

    # load dataset
    train_path = 'dataset/'
    trainset = TrainDataset(train_path, Vis_RGB, True, patch_size=128)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    # optimizer
    lr = 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    if model_load:
        optimizer.load_state_dict(torch.load(optimizer_path))
    fusion_loss = Swin_loss()

    # training
    print('Staring Training {} on {}'.format(method_type, device))
    epoch = 100
    for e in range(epoch):
        e += 0
        if e == epoch:
            break
        if e > 80:
            lr_new = lr / 100
            # modify learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_new
        elif e > 50:
            lr_new = lr / 10
            # modify learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_new
        loss1_total = 0.
        loss2_total = 0.
        loss3_total = 0.
        loss_total = 0.
        count = 0
        model.train()
        train_tqdm = tqdm(trainloader, total=len(trainloader))
        for vis_image, inf_image, p1,p2,p3,p4,p5, name in train_tqdm:
            count += 1
            vis_image = vis_image.to(device)
            inf_image = inf_image.to(device)
            p1 = p1.to(device)
            p2 = p2.to(device)
            p3 = p3.to(device)
            p4 = p4.to(device)
            p5 = p5.to(device)

            output = model(inf_image, vis_image)
            optimizer.zero_grad()
            total_loss, loss_text, loss_int, loss_ssim = fusion_loss(inf_image, vis_image, p1,p2,p3,p4,p5, output)
            total_loss.backward()
            optimizer.step()
            train_tqdm.set_postfix(epoch=e + 1, loss_text=loss_text.item(), loss_int=loss_int.item(),
                                   loss_ssim=loss_ssim.item(), total_loss=total_loss.item())
            loss1_total += loss_text
            loss2_total += loss_int
            loss3_total += loss_ssim
            loss_total += total_loss

        # Save the model once per epoch
        torch.save(model.state_dict(), f'{model_save_path}/{method_type}_epoch_{e + 1}.pth')
        torch.save(optimizer.state_dict(), f'{model_save_path}/optimizer_epoch_{e + 1}.pth')
        loss1_total /= count
        loss2_total /= count
        loss3_total /= count
        loss_total /= count
        print()
        print('Epoch {} | loss_text={} | loss_int={} | loss_ssim={} | loss_total={}'
              .format(e + 1, loss1_total, loss2_total, loss3_total, loss_total))
        print()

    print('Done.')


if __name__ == '__main__':
    main()
