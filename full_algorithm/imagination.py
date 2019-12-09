import abc
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class Imagination(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def generate(self, curr_goals, end_goals):
        pass

    @abc.abstractmethod
    def train_on_batch(self, curr_goals, end_goals, target_goals):
        pass


class ImaginationGAN(Imagination):
    def __init__(self, generator, discriminator, grad_penalty=False, diversity_sensitivity=True, lr=0.0001,
                 betas=(0.5,0.999), num_discrim_updates=1, l2_reg_d=0.0001, l2_reg_g=0.0001, ds_coeff=1.0,
                 gp_coeff=10.0, tau=5, device="cpu"):
        self.generator = generator
        self.discriminator = discriminator
        self.tau = tau
        self.goal_dim = generator.goal_dim
        self.target_dim = generator.out_dim
        self.latent_dim = generator.latent_dim
        self.grad_penalty = grad_penalty
        self.diversity_sensitivity = diversity_sensitivity
        self.num_discrim_updates = num_discrim_updates
        self.train_steps = 0
        self.l2_reg_g = l2_reg_g
        self.l2_reg_d = l2_reg_d
        self.ds_coeff = ds_coeff
        self.gp_coeff = gp_coeff
        self.optimiser_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.optimiser_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.cg_state_scaler = StandardScaler()
        self.goal_scaler = StandardScaler()
        self.device = device

    def fit_scaler(self, cg_states, goals):
        self.cg_state_scaler.fit(cg_states)
        self.goal_scaler.fit(goals)

    def load_params(self, g_dict, d_dict, cg_scaler, goal_scaler):
        if g_dict is not None:
            self.generator.load_state_dict(g_dict)
        if d_dict is not None:
            self.discriminator.load_state_dict(d_dict)
        if cg_scaler is not None:
            self.cg_state_scaler = cg_scaler
        if goal_scaler is not None:
            self.goal_scaler = goal_scaler

    def get_params(self):
        return self.generator.state_dict(), self.discriminator.state_dict(), self.cg_state_scaler, self.goal_scaler

    def generate(self, curr_goals, end_goals, eval=False):
        if len(curr_goals.shape) == 1:
            curr_goals = curr_goals.reshape(1, -1)
        if len(end_goals.shape) == 1:
            end_goals = end_goals.reshape(1, -1)
        curr_goals = torch.tensor(self.cg_state_scaler.transform(curr_goals), dtype=torch.float32, device=self.device)
        end_goals = torch.tensor(self.goal_scaler.transform(end_goals), dtype=torch.float32, device=self.device)
        z = torch.randn(curr_goals.shape[0], self.latent_dim, dtype=torch.float32, device=self.device)
        if eval:
            self.generator.eval()
        gen_out = self.generator(z, curr_goals, end_goals).detach().cpu().data.numpy()
        if eval:
            self.generator.train()
        return self.cg_state_scaler.inverse_transform(gen_out)

    def train_on_batch(self, curr_goals, end_goals, target_goals):
        batch_size = curr_goals.shape[0]
        curr_goals = torch.tensor(self.cg_state_scaler.transform(curr_goals), dtype=torch.float32, device=self.device)
        end_goals = torch.tensor(self.goal_scaler.transform(end_goals), dtype=torch.float32, device=self.device)
        target_goals = torch.tensor(self.cg_state_scaler.transform(target_goals), dtype=torch.float32, device=self.device)
        #curr_goals = torch.tensor(curr_goals, dtype=torch.float32, device=self.device)
        #end_goals = torch.tensor(end_goals, dtype=torch.float32, device=self.device)
        #target_goals = torch.tensor(target_goals, dtype=torch.float32, device=self.device)

        """TRAIN DISCRIMINATOR"""
        losses = {}
        emp_dists = (curr_goals-target_goals).norm(dim=-1)
        losses["emp_avg"] = torch.mean(emp_dists)
        losses["emp_std"] = torch.std(emp_dists)
        self.optimiser_d.zero_grad()
        z = torch.randn(batch_size, self.latent_dim, dtype=torch.float32, device=self.device)
        gen_target_goals = self.generator(z, curr_goals, end_goals)
        dists = (curr_goals-gen_target_goals).norm(dim=-1)
        losses["avg_dist"] = torch.mean(dists)
        losses["std_dist"] = torch.std(dists)
        d_real = self.discriminator(curr_goals, target_goals, end_goals)
        d_fake = self.discriminator(curr_goals, gen_target_goals, end_goals)
        loss_D = d_real.mean() - d_fake.mean()
        losses["loss_D"] = loss_D.item()
        loss_D_tot = loss_D
        if self.grad_penalty:
            alpha = torch.tensor(np.random.random((batch_size, 1)), dtype=torch.float32, device=self.device)
            interpolates = (alpha*target_goals + (1-alpha)*gen_target_goals)
            d_interpolates = self.discriminator(curr_goals, interpolates, end_goals)
            g_out = torch.ones(batch_size, 1, dtype=torch.float32, device=self.device)
            gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_output=g_out, create_graph=True,
                                      retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0),-1)
            grad_loss = self.gp_coeff*((gradients.norm(2, dim=1)-1)**2).mean()
            loss_D_tot = loss_D_tot + grad_loss
            losses["loss_GP"] = grad_loss.item()
        loss_D_reg = 0
        for p in self.discriminator.parameters():
            loss_D_reg += torch.dot(p.view(-1), p.view(-1))
        loss_D_reg = loss_D_reg*self.l2_reg_d
        losses["loss_D_reg"] = loss_D_reg.item()
        loss_D_tot = loss_D_tot + loss_D_reg
        losses["loss_D_tot"] = loss_D_tot.item()
        loss_D_tot.backward()
        self.optimiser_d.step()

        if self.train_steps % self.num_discrim_updates == 0:
            """TRAIN GENERATOR"""
            self.optimiser_g.zero_grad()
            z2 = torch.randn(batch_size, self.latent_dim, dtype=torch.float32, device=self.device)
            gen_target_goals = self.generator(z2, curr_goals, end_goals)
            d_fake = self.discriminator(curr_goals, gen_target_goals, end_goals)
            loss_G = d_fake.mean()
            losses["loss_G"] = loss_G.item()
            loss_G_tot = loss_G
            if self.diversity_sensitivity:
                z_alt = torch.randn(batch_size, self.latent_dim, dtype=torch.float32, device=self.device)
                gen_target_goals_alt = self.generator(z_alt, curr_goals, end_goals).detach()
                state_diff_l1 = F.l1_loss(gen_target_goals, gen_target_goals_alt, reduce=False).sum(dim=-1) / self.target_dim
                z_diff_l1 = F.l1_loss(z.detach(), z_alt.detach(), reduce=False).sum(dim=-1) / self.latent_dim
                ds_loss = -1*self.ds_coeff*(state_diff_l1 / (z_diff_l1+1e-5)).mean()
                losses["loss_DS"] = ds_loss.item()
                loss_G_tot = loss_G_tot + ds_loss
            loss_G_reg = 0
            for p in self.generator.parameters():
                loss_G_reg += torch.dot(p.view(-1), p.view(-1))
            loss_G_reg = loss_G_reg*self.l2_reg_g
            losses["loss_G_reg"] = loss_G_reg.item()
            loss_G_tot = loss_G_tot + loss_G_reg
            losses["loss_G_tot"] = loss_G_tot.item()
            loss_G_tot.backward()
            self.optimiser_g.step()

        return losses


