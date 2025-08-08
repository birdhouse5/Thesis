import torch
import torch.nn.functional as F
from torch.optim import Adam

class Trainer:
    def __init__(self, env, vae, policy, cfg):
        self.env, self.vae, self.policy = env, vae, policy
        self.device = torch.device(cfg.device)
        self.vae_opt = Adam(vae.parameters(), lr=cfg.vae_lr)
        self.pol_opt = Adam(policy.parameters(), lr=cfg.policy_lr)
        self.gamma, self.lam = 0.99, 0.95
        self.episode_count = 0
        self.total_steps = 0

    def train_episode(self):
        obs = torch.tensor(self.env.reset(), dtype=torch.float32, device=self.device).unsqueeze(0)
        actions_seq, rewards_seq = [], []
        traj = {k: [] for k in ['obs', 'acts', 'rews', 'logp', 'vals', 'lat']}
        done = False

        while not done:
            if actions_seq:
                obs_seq = torch.stack(traj['obs']).unsqueeze(0)
                act_seq = torch.stack(actions_seq).unsqueeze(0)
                rew_seq = torch.tensor(rewards_seq, device=self.device).unsqueeze(0).unsqueeze(-1)
                latent, _, _, _ = self.vae(obs_seq, act_seq, rew_seq)
            else:
                latent = torch.zeros((1, self.vae.latent_dim), device=self.device)

            act_dict, val = self.policy.act(obs, latent)
            weights = act_dict['long_weights'][0]
            next_obs, r, done, _ = self.env.step(weights)

            logp = torch.log(act_dict['decision_probs'] + 1e-8).mean()
            traj['obs'].append(obs.squeeze(0))
            traj['acts'].append(weights)
            traj['rews'].append(torch.tensor(r, device=self.device))
            traj['logp'].append(logp)
            traj['vals'].append(val.squeeze())
            traj['lat'].append(latent.squeeze())

            actions_seq.append(weights)
            rewards_seq.append([r])

            obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.total_steps += 1

        adv, gae = [], 0
        rews = torch.stack(traj['rews'])
        vals = torch.stack(traj['vals'])
        for t in reversed(range(len(rews))):
            delta = rews[t] + self.gamma * (vals[t + 1] if t + 1 < len(vals) else 0) - vals[t]
            gae = delta + self.gamma * self.lam * gae
            adv.insert(0, gae)
        adv = torch.tensor(adv, device=self.device)
        ret = adv + vals

        logp = torch.stack(traj['logp'])
        pol_loss = -(adv.detach() * logp).mean()
        val_loss = F.mse_loss(vals, ret.detach())

        self.pol_opt.zero_grad()
        (pol_loss + 0.5 * val_loss).backward()
        self.pol_opt.step()

        obs_seq = torch.stack(traj['obs']).unsqueeze(0)
        act_seq = torch.stack(traj['acts']).unsqueeze(0)
        rew_seq = torch.tensor(rewards_seq, device=self.device).unsqueeze(0)
        vae_loss, _ = self.vae.compute_loss(obs_seq, act_seq, rew_seq)

        self.vae_opt.zero_grad()
        vae_loss.backward()
        self.vae_opt.step()

        self.episode_count += 1
        return rews.sum().item(), vae_loss.item(), pol_loss.item()
