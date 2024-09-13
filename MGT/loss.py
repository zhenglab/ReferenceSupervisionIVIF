import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


##################################################################
#                            ssim
##################################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def Contrast(img1, img2, window_size=11, channel=1):
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


##################################################################
#                     SwinFusion loss
##################################################################
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = func.conv2d(x, self.weightx, padding=1)
        sobely = func.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()
    def forward(self, image_A, image_B, p1,p2,p3,p4,p5, image_fused):
        gradient_1 = self.sobelconv(p1)
        gradient_2 = self.sobelconv(p2)
        gradient_3 = self.sobelconv(p3)
        gradient_4 = self.sobelconv(p4)
        gradient_5 = self.sobelconv(p5)
        gradient_fused = self.sobelconv(image_fused)
        Loss_gradient = func.l1_loss(gradient_fused, gradient_1)+\
                        func.l1_loss(gradient_fused, gradient_2)+\
                        func.l1_loss(gradient_fused, gradient_3)+\
                        func.l1_loss(gradient_fused, gradient_4)+\
                        func.l1_loss(gradient_fused, gradient_5)
        return Loss_gradient

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, p1,p2,p3,p4,p5, image_fused):
        gradient_1 = self.sobelconv(p1)
        gradient_2 = self.sobelconv(p2)
        gradient_3 = self.sobelconv(p3)
        gradient_4 = self.sobelconv(p4)
        gradient_5 = self.sobelconv(p5)
        Loss_SSIM = ssim(image_fused, gradient_1)+\
                    ssim(image_fused, gradient_2)+\
                    ssim(image_fused, gradient_3)+\
                    ssim(image_fused, gradient_4)+\
                    ssim(image_fused, gradient_5)
        return Loss_SSIM

def L_Intensity(image_A, image_B, p1,p2,p3,p4,p5, image_fused):
    Loss_intensity = func.l1_loss(image_fused, p1)
    Loss_intensity = func.l1_loss(image_fused, p2)
    Loss_intensity = func.l1_loss(image_fused, p3)
    Loss_intensity = func.l1_loss(image_fused, p4)
    Loss_intensity = func.l1_loss(image_fused, p5)        
    return Loss_intensity

class SwinFusion_loss(nn.Module):
    def __init__(self):
        super(SwinFusion_loss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity
        self.L_SSIM = L_SSIM()

        # print(1)
    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 50 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 50 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

class SwinFusion_loss(nn.Module):
    def __init__(self):
        super(SwinFusion_loss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, p1,p2,p3,p4,p5, image_fused):
        loss_l1 = 50 * self.L_Inten(image_A, image_B, p1,p2,p3,p4,p5, image_fused)
        loss_gradient = 50 * self.L_Grad(image_A, image_B, p1,p2,p3,p4,p5, image_fused)
        loss_SSIM = 10 * (1 - self.L_SSIM(image_A, image_B, p1,p2,p3,p4,p5, image_fused))
        fusion_loss = loss_l1 + loss_gradient + loss_SSIM
        return fusion_loss, loss_gradient, loss_l1, loss_SSIM

