from filtering import compute_graph
import torch
ch = torch
import numpy as np
import torch.nn.functional as F
import kornia


class Surf(ch.nn.Module):
    def __init__(self, alpha=50, th=0.002, image_dim=32):
        super(Surf, self).__init__()
        self.alpha = alpha
        self.th = th
        self.image_dim = image_dim
    
    def forward(self, x):
#         d_up = (x - ch.roll(x, -1, 2)).pow(2).sum(dim=1)#.sqrt()
#         d_right = (x - ch.roll(x, 1, 3)).pow(2).sum(dim=1)#.sqrt()
        d_up = (x - ch.roll(x, -1, 2)).abs().sum(dim=1)#.sqrt()
        d_right = (x - ch.roll(x, 1, 3)).abs().sum(dim=1)#.sqrt()
        d_up = 1 / (1 + ch.exp(-self.alpha * (d_up - self.th)))
        d_right = 1 / (1 + ch.exp(-self.alpha * (d_right - self.th)))
        d = (d_up + d_right) / 2
        return d.view(d.size(0), -1).sum(dim=1) / (self.image_dim**2)

# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        # self.surf_term = torch.nn.MSELoss(reduction='mean')
        self.surf_mod = Surf(image_dim=noisy_image.shape[-1])
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        gamma = 0.96
        # clean_image = torch.sigmoid(self.clean_image)
        clean_image = self.clean_image
#         target = compute_surf_energy(x.unsqueeze(0), 200)
        tv_loss = self.regularization_term(clean_image) / (clean_image.shape[2] * clean_image.shape[3])
        # surf_loss = self.surf_mod(clean_image.unsqueeze(0))
        surf_loss = self.surf_mod(clean_image)
        prior_loss = self.l2_term(clean_image, self.noisy_image)
        return (1 - gamma) * prior_loss + gamma * 0.1*(tv_loss+ 50*surf_loss)
        
    def get_clean_image(self):
        # return torch.sigmoid(self.clean_image)
        return self.clean_image


def denoiser(optimizer, tv_denoiser):
    # run the optimization loop
    num_iters = 80
    for i in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser().mean()
        # if i % 500 == 0:
        #     print("Loss in iteration {} of {}: {:.3f}".format(i, num_iters, loss.item()))
        loss.backward()
        optimizer.step()
        tv_denoiser.clean_image.data = torch.clamp(tv_denoiser.clean_image.data,0,1)
    return tv_denoiser.get_clean_image().detach()


def denoise(im, args):
    tv_denoiser = TVDenoise(im).cuda()
    # define the optimizer to optimize the 1 parameter of tv_denoiser
    optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=args.batch_size * 2.5 / 15, momentum=0.9)
    imd = denoiser(optimizer, tv_denoiser)
    return imd


def compute_graph_torch(im, args, upsample=False):
    dim = args.dim if args.dim and upsample else im.shape[-1]
    scale_factor = args.dim / im.shape[-1]
    # if upsample:
    #     im = ch.nn.functional.interpolate(im, scale_factor=scale_factor)
    labelmat = None
    rec_img, labelmat = compute_graph((im.cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8), 
                    dim, dim, args.res, False,im.shape[0], "/home/matteo/testdir", 0, False)
    rec_img = ch.from_numpy(rec_img).cuda().permute(0,3,1,2) / 255.
    # if upsample:
    #     rec_img = ch.nn.functional.interpolate(rec_img, scale_factor=1/scale_factor)
    return rec_img, labelmat


def l2_project(orig_input, x, eps):
    diff = x - orig_input
    diff = diff.renorm(p=2, dim=0, maxnorm=eps)
    return ch.clamp(orig_input + diff, 0, 1)


def linf_project(orig_input, x, eps):
    diff = x - orig_input
    diff = ch.clamp(diff, -eps, eps)
    return ch.clamp(diff + orig_input, 0, 1)


def BPDA(im, target, model, args, attack_kwargs, do_denoise):
    eps = attack_kwargs['eps']
    if eps > 0:
        if attack_kwargs['constraint'] == '2':
            project = l2_project
            l = len(im.shape) - 1
            rp = ch.randn_like(im)
            rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
            im = ch.clamp(im + eps * rp / (rp_norm + 1e-10), 0, 1)

        elif attack_kwargs['constraint'] == 'inf':
            project = linf_project
            im = im.detach() + torch.zeros_like(im).uniform_(-eps, eps)
            im = torch.clamp(im, 0, 1)

    if do_denoise:
        print('Doing image denoising')
        im = denoise(im, args)
    
    make_adv = False
    if attack_kwargs["eps"] > 0:
        make_adv = True

    best_loss = None
    best_x = None

    if args.rec_first:
        im, _ = compute_graph_torch(im, args, upsample=False)

    # A function that updates the best loss and best input
    def replace_best(loss, bloss, x, bx):
        if bloss is None:
            bx = x.clone().detach()
            bloss = loss.clone().detach()
        else:
            replace = bloss < loss
            bx[replace] = x[replace].clone().detach()
            bloss[replace] = loss[replace]
        return bloss, bx
    adv_im = im.detach().clone()

    if make_adv:
        for j in range(args.outer_attack_steps):
            adv_im, _ = compute_graph_torch(adv_im, args, upsample=False)
            adv_out, adv_im = model(adv_im, target, make_adv, **attack_kwargs)
            with ch.no_grad():
                adv_im = project(im, adv_im, args.eps)
                losses = F.cross_entropy(adv_out, target, reduce=False)
            # print(losses.mean().item())
                best_loss, best_x = replace_best(losses, best_loss, adv_im, best_x)

        best_x, _ = compute_graph_torch(best_x, args, upsample=False)
        # best_x = project(im, best_x, args.eps)
    else:
        if not args.rec_first:
            best_x, _ = compute_graph_torch(adv_im, args, upsample=False)
            # best_x = im
        else:
            best_x = im
    return best_x

import eagerpy as ep
