import torch
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class ImaginationModule:
    def __init__(self, G_nets, D_nets, G_opts, D_opts, one_step_model, latent_dim, tau=5, l2_reg_D=0.0, l2_reg_G=0.0,
                 reg_with_osm=True, l2_loss_coeff=30.0, use_all_osms_for_each_gan=False, device="cpu"):
        self.G_nets = G_nets
        self.D_nets = D_nets
        self.G_opts = G_opts
        self.D_opts = D_opts
        self.one_step_model = one_step_model
        self.l2_reg_D = l2_reg_D
        self.l2_reg_G = l2_reg_G
        self.reg_with_osm = reg_with_osm
        self.l2_loss_coeff = l2_loss_coeff
        self.tau = tau
        self.latent_dim = latent_dim
        self.state_dim = G_nets[0].state_dim
        self.ac_dim = G_nets[0].ac_dim
        self.goal_dim = G_nets[0].goal_dim
        self.goal_scaler = StandardScaler()
        self.use_all_osms_for_each_gan = use_all_osms_for_each_gan
        self.device = device

    def test_traj_rand_gan(self, start_states, end_goals, num_steps=None, frac_replaced_with_osm_pred=0.0):
        num_gans = len(self.G_nets)
        for i in range(num_gans):
            self.G_nets[i].eval()
        if len(start_states.shape) == 1:
            start_states = start_states[np.newaxis, ...]
            if len(end_goals.shape) > 1:
                if end_goals.shape[0] != 1:
                    raise ValueError()
            else:
                end_goals = end_goals[np.newaxis, ...]
        if num_steps is None:
            num_steps = self.tau
        start_states = self.one_step_model.networks[0].state_scaler.transform(start_states)
        end_goals = self.goal_scaler.transform(end_goals)
        start_states = torch.tensor(start_states, dtype=torch.float32, device=self.device)
        end_goals = torch.tensor(end_goals, dtype=torch.float32, device=self.device)
        gen_states = np.zeros((start_states.size()[0], num_steps+1, self.state_dim))
        gen_actions = np.zeros((start_states.size()[0], num_steps, self.ac_dim))
        gen_states[:, 0, :] = start_states.cpu().data.numpy()
        if len(end_goals.shape) == 2:
            end_goals = end_goals.unsqueeze(0).repeat(num_steps, 1, 1).transpose(0,1)
        states = start_states
        for ts in range(num_steps):
            z = torch.randn(start_states.size()[0], self.latent_dim, dtype=torch.float32, device=self.device)
            inds = np.array_split(np.random.permutation(start_states.size()[0]), num_gans)
            for i in range(num_gans):
                a, s, _ = self.G_nets[i].one_step_generator(z[inds[i], ...], states[inds[i], ...], end_goals[inds[i], ts, :])
                gen_states[inds[i], ts+1, :] = s.cpu().data.numpy()
                gen_actions[inds[i], ts, :] = a.cpu().data.numpy()
            if frac_replaced_with_osm_pred > 0.0:
                num_osms = len(self.one_step_model.networks)
                n_inds = int(gen_states.shape[0]*frac_replaced_with_osm_pred)
                inds = np.array_split(np.random.permutation(gen_states.shape[0])[:n_inds], num_osms)
                for i in range(num_osms):
                    gen_states[inds[i], ts+1, :] = self.one_step_model.predict(gen_states[inds[i], ts, :], gen_actions[inds[i], ts, :],
                                                                               osm_ind=i, normed_input=True)
            states = torch.tensor(gen_states[:, ts+1, :], dtype=torch.float32, device=self.device)

        gen_states = self.one_step_model.networks[0].state_scaler.inverse_transform(
            gen_states.reshape(-1, self.state_dim)).reshape(start_states.size()[0], num_steps + 1,
                                                                               self.state_dim)
        return gen_states, gen_actions

    def train_on_trajs(self, states, end_goals, actions, object=False, gan_ind=0):
        """
        states should be [batch x (tau+1) x state_dim]
        end_goals: [batch x goal_dim]
        actions: [batch x tau x ac_dim]

        object: just for monitoring distances in testing.
        """
        self.G_nets[gan_ind].train()
        self.D_nets[gan_ind].train()
        batch_size = states.shape[0]
        real_states = torch.tensor(self.one_step_model.networks[0].state_scaler.transform(states.reshape(-1, self.state_dim)).reshape(batch_size, self.tau+1, self.state_dim),
                                   dtype=torch.float32, device=self.device)
        end_goals = torch.tensor(self.goal_scaler.transform(end_goals.reshape(-1, self.goal_dim)).reshape(batch_size, self.tau, self.goal_dim),
                                 dtype=torch.float32, device=self.device)
        real_actions = torch.tensor(actions, dtype=torch.float32, device=self.device)

        losses = {}
        """Train discriminator"""
        self.D_opts[gan_ind].zero_grad()
        z = torch.randn(batch_size, self.latent_dim, dtype=torch.float32, device=self.device)
        gen_states, gen_actions, _ = self.G_nets[gan_ind](z, real_states[:, 0, :], end_goals)
        d_real = self.D_nets[gan_ind](real_states[:, :-1, :], end_goals, real_states[:, 1:, :],
                           real_actions).squeeze().mean(dim=-1)
        d_fake = self.D_nets[gan_ind](gen_states[:, :-1, :], end_goals, gen_states[:, 1:, :],
                           gen_actions).squeeze().mean(dim=-1)
        loss_D_main = d_real.mean() - d_fake.mean()
        losses["loss_D_main"] = loss_D_main.item()
        loss_D_reg = 0
        for par in self.D_nets[gan_ind].parameters():
            loss_D_reg += torch.dot(par.view(-1), par.view(-1))
        loss_D_reg = self.l2_reg_D*loss_D_reg
        losses["loss_D_reg"] = loss_D_reg.item()
        loss_D_tot = loss_D_main + loss_D_reg
        loss_D_tot.backward()
        self.D_opts[gan_ind].step()

        """monitor final distance - just for fetch envs - delete if/when working"""
        emp_dist = (real_states[:, 0, :3] - real_states[:, -1, :3]).norm(dim=-1)
        gen_dist = (gen_states[:, 0, :3] - gen_states[:, -1, :3]).norm(dim=-1)
        losses["emp_dist"] = emp_dist.mean()
        losses["emp_std"] = emp_dist.std()
        losses["gen_dist"] = gen_dist.mean()
        losses["gen_std"] = gen_dist.std()
        if object:
            emp_obj_dist = (real_states[:, 0, 3:6] - real_states[:, -1, 3:6]).norm(dim=-1)
            gen_obj_dist = (gen_states[:, 0, 3:6] - gen_states[:, -1, 3:6]).norm(dim=-1)
            losses["emp_obj_dist"] = emp_obj_dist.mean()
            losses["emp_obj_std"] = emp_obj_dist.std()
            losses["gen_obj_dist"] = gen_obj_dist.mean()
            losses["gen_obj_std"] = gen_obj_dist.std()

        """Train Generator"""
        self.G_opts[gan_ind].zero_grad()
        z_2 = torch.randn(batch_size, self.latent_dim, dtype=torch.float32, device=self.device)
        gen_states, gen_actions, z_all = self.G_nets[gan_ind](z_2, real_states[:, 0, :], end_goals)
        d_fake = self.D_nets[gan_ind](gen_states[:, :-1, :], end_goals, gen_states[:, 1:, :],
                           gen_actions).squeeze().mean(dim=-1)
        loss_G_main = d_fake.mean()
        losses["loss_G_main"] = loss_G_main.item()
        loss_G_tot = loss_G_main
        if self.reg_with_osm:
            diff_unnorm = self.one_step_model.networks[0].scaler_transform(gen_states[:, 1:, :], "state", inverse=True) - \
                          self.one_step_model.networks[0].scaler_transform(gen_states[:, :-1, :], "state", inverse=True)
            diff = self.one_step_model.networks[0].scaler_transform(diff_unnorm, "diff", inverse=False)
            num_osms = len(self.one_step_model.networks)
            if self.use_all_osms_for_each_gan:
                pred_diff = torch.zeros(num_osms, *diff.shape, device=self.device)
                for j in range(num_osms):
                    pred_diff[j, ...] = self.one_step_model.networks[j](gen_states[:, :-1, :], gen_actions)
                pred_diff = torch.mean(pred_diff, dim=0)
            else:
                pred_diff = self.one_step_model.networks[gan_ind](gen_states[:, :-1, :], gen_actions)
            l2_loss = F.mse_loss(pred_diff, diff)
            l2_loss *= self.l2_loss_coeff
            loss_G_tot = loss_G_tot + l2_loss
            losses["loss_model_l2"] = l2_loss.item()
        #parameter l2 loss
        loss_G_reg = 0
        for par in self.G_nets[gan_ind].parameters():
            loss_G_reg += torch.dot(par.view(-1), par.view(-1))
        loss_G_reg = loss_G_reg * self.l2_reg_G
        loss_G_tot = loss_G_tot + loss_G_reg
        losses["loss_G_reg"] = loss_G_reg.item()
        loss_G_tot.backward()
        self.G_opts[gan_ind].step()

        return losses

    def fit_scalers(self, goals, noise=0.05):
        self.goal_scaler.fit(goals+noise*np.random.randn(*goals.shape))

    def load(self, dictGs, dictDs, goal_scaler=None, one_step=None):
        if dictGs is not None:
            for i, dict in enumerate(dictGs):
                self.G_nets[i].load_state_dict(dict)
        if dictDs is not None:
            for i, dict in enumerate(dictDs):
                self.D_nets[i].load_state_dict(dict)
        if goal_scaler is not None:
            self.goal_scaler = goal_scaler
        if one_step is not None:
            self.one_step_model.load(*one_step)

    def save(self, one_step_also=True):
        res = [net.state_dict() for net in self.G_nets], [net.state_dict() for net in self.D_nets], self.goal_scaler
        if one_step_also:
            return res, self.one_step_model.save()
        else:
            return res