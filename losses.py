import torch
import torch.nn.functional as F
import pdb

# DCGAN loss
def loss_dcgan_dis(dis_fake, dis_real):
  L1 = torch.mean(F.softplus(-dis_real))
  L2 = torch.mean(F.softplus(dis_fake))
  return L1, L2


def loss_dcgan_gen(dis_fake, M_regu=None):
  loss = torch.mean(F.softplus(-dis_fake))
  return loss


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
  loss_real = torch.mean(F.relu(1. - dis_real))
  loss_fake = torch.mean(F.relu(1. + dis_fake))
  return loss_real, loss_fake
# def loss_hinge_dis(dis_fake, dis_real): # This version returns a single loss
  # loss = torch.mean(F.relu(1. - dis_real))
  # loss += torch.mean(F.relu(1. + dis_fake))
  # return loss


def loss_hinge_gen(dis_fake, M_regu=None):
  loss = -torch.mean(dis_fake)
  if M_regu is not None:
       #loss_M = torch.mean(F.mse_loss(M_regu[0], M_regu[1])) + torch.mean(F.mse_loss(M_regu[2], M_regu[3]))
       loss =  loss  
  return loss

# Default to hinge loss
 #generator_loss = loss_hinge_gen
 #discriminator_loss = loss_hinge_dis
generator_loss = loss_dcgan_gen
discriminator_loss = loss_dcgan_dis
