# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Takeyuki-K
"""PPO for K1-MP. Stage 1: motion imitation. Stage 2: RL fine-tune (stand<->walk, DR, pushes, energy).
usage: python3 ppo.py --stage 1 --iters 1500 --out runs/s1
       python3 ppo.py --stage 2 --iters 3000 --init runs/s1/model.pt --out runs/s2
"""
import argparse, os, time, json
import numpy as np
import torch
import torch.nn as nn
from k1env import K1Batch
from k1env_eco import K1EcoBatch
from k1env_speed import K1SpeedBatch

torch.set_num_threads(2)


class RunningNorm(nn.Module):
    def __init__(self, n, clip=5.0):
        super().__init__()
        self.register_buffer('mean', torch.zeros(n)); self.register_buffer('var', torch.ones(n))
        self.register_buffer('count', torch.tensor(1e-4)); self.clip = clip

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


def mlp(i, o, h=(512, 256, 128)):
    L = []
    for k in h:
        L += [nn.Linear(i, k), nn.ELU()]; i = k
    L.append(nn.Linear(i, o))
    return nn.Sequential(*L)


class AC(nn.Module):
    def __init__(self, na, nc, nact):
        super().__init__()
        self.na_norm = RunningNorm(na); self.nc_norm = RunningNorm(nc)
        self.actor = mlp(na, nact); self.critic = mlp(nc, 1)
        self.log_std = nn.Parameter(torch.full((nact,), -1.0))
        with torch.no_grad():
            self.actor[-1].weight.mul_(0.01); self.actor[-1].bias.zero_()

    def dist(self, oa):
        mu = self.actor(self.na_norm(oa))
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))

    def value(self, oc):
        return self.critic(self.nc_norm(oc)).squeeze(-1)


ACEco = AC


def transfer_fixed_to_eco(src, net):
    """12-action fixed-gain policy -> 36-action eco policy. New obs columns (last_a 12->36) get zero
    weights, new outputs (Kp/Kd scales) start at 0 => scale 1 => identical behaviour at iteration 0."""
    dst = net.state_dict()
    na_old = src['na_norm.mean'].shape[0]; nc_old = src['nc_norm.mean'].shape[0]
    la_old = na_old  # actor obs: [..., last_a(12)] at the end of actor part
    # actor obs layout old: base(34) + last_a(12) ; new: base(34) + last_a(36)
    base = na_old - 12
    def map_cols(old_w, new_w, n_old_total):
        new_w.zero_()
        new_w[:, :base + 12] = old_w[:, :base + 12]            # base + old last_a (position part)
        new_w[:, base + 36:] = old_w[:, base + 12:n_old_total] # critic privileged part (if any)
        return new_w
    with torch.no_grad():
        for k in dst:
            if k.startswith('na_norm') or k.startswith('nc_norm'):
                if k.endswith('count'):
                    dst[k] = src[k].clone(); continue
                o = src[k]; nw = torch.zeros_like(dst[k]) if k.endswith('mean') else torch.ones_like(dst[k])
                nw[:base + 12] = o[:base + 12]; nw[base + 36:] = o[base + 12:]
                dst[k] = nw
            elif k == 'actor.0.weight':
                dst[k] = map_cols(src[k], dst[k].clone(), na_old)
            elif k == 'critic.0.weight':
                dst[k] = map_cols(src[k], dst[k].clone(), nc_old)
            elif k == f'actor.{len(net.actor) - 1}.weight':
                w = torch.zeros_like(dst[k]); w[:12] = src[k]; dst[k] = w
            elif k == f'actor.{len(net.actor) - 1}.bias':
                b = torch.zeros_like(dst[k]); b[:12] = src[k]; dst[k] = b
            elif k == 'log_std':
                ls = torch.full_like(dst[k], np.log(0.25)); ls[:12] = src[k]; dst[k] = ls
            else:
                dst[k] = src[k].clone()
    net.load_state_dict(dst)


def transfer_eco_to_speed(src, net, at=8, add=2):
    """eco policy (obs 70) -> speed policy (obs 72): insert `add` zero-weight inputs at index `at`
    (v_cmd, v_ref after alpha). Behaviour at iteration 0 is identical to the eco policy."""
    dst = net.state_dict()
    with torch.no_grad():
        for k in dst:
            o = src[k]
            if k.startswith(('na_norm', 'nc_norm')) and not k.endswith('count'):
                fill = torch.zeros(add) if k.endswith('mean') else torch.ones(add)
                if k.endswith('mean'):
                    fill = torch.full((add,), 0.8)
                    fill_var = None
                dst[k] = torch.cat([o[:at], fill if k.endswith('mean') else torch.full((add,), 0.1), o[at:]])
            elif k in ('actor.0.weight', 'critic.0.weight'):
                dst[k] = torch.cat([o[:, :at], torch.zeros(o.shape[0], add), o[:, at:]], 1)
            else:
                dst[k] = o.clone()
    net.load_state_dict(dst)


def transfer_insert_obs(src, net, at, add=1):
    """insert `add` zero-weight observation inputs at index `at` (actor and critic)."""
    dst = net.state_dict()
    with torch.no_grad():
        for k in dst:
            o = src[k]
            if k.startswith(('na_norm', 'nc_norm')) and not k.endswith('count'):
                dst[k] = torch.cat([o[:at], torch.zeros(add) if k.endswith('mean') else torch.full((add,), 0.1), o[at:]])
            elif k in ('actor.0.weight', 'critic.0.weight'):
                dst[k] = torch.cat([o[:, :at], torch.zeros(o.shape[0], add), o[:, at:]], 1)
            else:
                dst[k] = o.clone()
    net.load_state_dict(dst)


def transfer_add_action(src, net, at=72):
    """36-action speed policy -> 37 actions (gait-speed choice). New last_action input (actor index 72) and
    new output start at zero => identical behaviour at iteration 0 (gait speed = commanded speed)."""
    dst = net.state_dict()
    with torch.no_grad():
        for k in dst:
            o = src[k]
            if k.startswith(('na_norm', 'nc_norm')) and not k.endswith('count'):
                dst[k] = torch.cat([o[:at], torch.zeros(1) if k.endswith('mean') else torch.full((1,), 0.1), o[at:]])
            elif k in ('actor.0.weight', 'critic.0.weight'):
                dst[k] = torch.cat([o[:, :at], torch.zeros(o.shape[0], 1), o[:, at:]], 1)
            elif k == f'actor.{len(net.actor) - 1}.weight':
                dst[k] = torch.cat([o, torch.zeros(1, o.shape[1])], 0)
            elif k == f'actor.{len(net.actor) - 1}.bias':
                dst[k] = torch.cat([o, torch.zeros(1)])
            elif k == 'log_std':
                dst[k] = torch.cat([o, torch.full((1,), float(np.log(0.3)))])
            else:
                dst[k] = o.clone()
    net.load_state_dict(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, default=1)
    ap.add_argument('--iters', type=int, default=1500)
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--horizon', type=int, default=32)
    ap.add_argument('--init', default='')
    ap.add_argument('--out', default='runs/s1')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--std_reset', type=float, default=0.0)
    ap.add_argument('--w_energy', type=float, default=0.0015)
    ap.add_argument('--from_fixed', action='store_true', help='init from a fixed-gain (12-action) policy')
    ap.add_argument('--from_eco', action='store_true', help='init from the eco (single-speed) policy')
    ap.add_argument('--seed', type=int, default=207)
    ap.add_argument('--lr_min', type=float, default=1e-5)
    ap.add_argument('--add_heading_obs', action='store_true', help='init from policy without heading-error input')
    ap.add_argument('--from_speed36', action='store_true', help='init from a 36-action speed policy (adds gait-speed action)')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = K1SpeedBatch(args.n, stage=2, randomize=True, seed=args.seed, w_energy=args.w_energy)
    oa, oc = env.obs()
    net = ACEco(oa.shape[1], oc.shape[1], env.nact)
    if args.init:
        sd = torch.load(args.init, map_location='cpu')
        if args.add_heading_obs:
            transfer_insert_obs(sd['model'], net, at=10)
        elif args.from_speed36:
            transfer_add_action(sd['model'], net)
        elif args.from_eco:
            transfer_eco_to_speed(sd['model'], net)
        elif args.from_fixed:
            transfer_fixed_to_eco(sd['model'], net)
        else:
            net.load_state_dict(sd['model'])
        print('loaded', args.init)
        if args.std_reset > 0:
            with torch.no_grad():
                net.log_std.fill_(np.log(args.std_reset))
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr = args.lr
    gamma, lam, clip = 0.99, 0.95, 0.2
    N, H = args.n, args.horizon
    ep_ret = np.zeros(N); ep_len = np.zeros(N)
    done_ret, done_len = [], []
    log = open(os.path.join(args.out, 'log.txt'), 'a')
    t0 = time.time(); steps = 0
    for it in range(args.iters):
        OA, OC, A, LP, R, D, V = [], [], [], [], [], [], []
        infos = {}
        net.eval()
        for h in range(H):
            ta, tc = torch.from_numpy(oa), torch.from_numpy(oc)
            with torch.no_grad():
                dist = net.dist(ta); a = dist.sample(); lp = dist.log_prob(a).sum(-1); v = net.value(tc)
            rew, term, trunc, info = env.step(a.numpy().astype(np.float64))
            for k in ('r_q', 'r_ank', 'r_vel', 'r_up', 'r_contact', 'hs_bad', 'vx', 'verr', 'v_cmd', 'head_err', 'v_lib_ratio', 'P_elec', 'kp_mean', 'kd_mean'):
                infos.setdefault(k, []).append(float(np.mean(info[k])))
            rew = rew.copy()
            if trunc.any():  # bootstrap on time-out
                _, oc_end = env.obs()
                with torch.no_grad():
                    vend = net.value(torch.from_numpy(oc_end)).numpy()
                rew = np.where(trunc & ~term, rew + gamma * vend, rew)
            ep_ret += rew; ep_len += 1
            done = term | trunc
            for i in np.where(done)[0]:
                done_ret.append(ep_ret[i]); done_len.append(ep_len[i])
            ep_ret[done] = 0; ep_len[done] = 0
            env.reset(np.where(done)[0])
            OA.append(ta); OC.append(tc); A.append(a); LP.append(lp); V.append(v)
            R.append(torch.from_numpy(rew)); D.append(torch.from_numpy(done.astype(np.float32)))
            oa, oc = env.obs()
        steps += N * H
        with torch.no_grad():
            v_last = net.value(torch.from_numpy(oc))
        OA = torch.stack(OA); OC = torch.stack(OC); A = torch.stack(A); LP = torch.stack(LP)
        V = torch.stack(V); R = torch.stack(R).float(); D = torch.stack(D)
        adv = torch.zeros_like(R); last = torch.zeros(N)
        for h in reversed(range(H)):
            nv = v_last if h == H - 1 else V[h + 1]
            delta = R[h] + gamma * nv * (1 - D[h]) - V[h]
            last = delta + gamma * lam * (1 - D[h]) * last
            adv[h] = last
        ret = adv + V
        net.na_norm.update(OA.reshape(-1, OA.shape[-1])); net.nc_norm.update(OC.reshape(-1, OC.shape[-1]))
        fa, fc, fA, fLP, fadv, fret, fV = (x.reshape(N * H, -1).squeeze(-1) if x.dim() == 2 else x.reshape(N * H, -1)
                                            for x in (OA, OC, A, LP, adv, ret, V))
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        net.train()
        nb = 4; bs = N * H // nb
        kls = []
        for ep in range(5):
            perm = torch.randperm(N * H)
            for b in range(nb):
                idx = perm[b * bs:(b + 1) * bs]
                dist = net.dist(fa[idx]); lp = dist.log_prob(fA[idx]).sum(-1)
                ratio = torch.exp(lp - fLP[idx])
                s1 = ratio * fadv[idx]; s2 = torch.clamp(ratio, 1 - clip, 1 + clip) * fadv[idx]
                v = net.value(fc[idx])
                vcl = fV[idx] + torch.clamp(v - fV[idx], -clip, clip)
                vloss = torch.max((v - fret[idx]) ** 2, (vcl - fret[idx]) ** 2).mean()
                loss = -torch.min(s1, s2).mean() + 1.0 * vloss - 0.002 * dist.entropy().sum(-1).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                with torch.no_grad():
                    kl = (fLP[idx] - lp).mean().item(); kls.append(kl)
            if np.mean(kls[-nb:]) > 0.03:
                break
        kl = np.mean(kls)
        lr = max(args.lr_min, lr / 1.5) if kl > 0.02 else (min(1e-3, lr * 1.5) if kl < 0.005 else lr)
        for g in opt.param_groups:
            g['lr'] = lr
        with torch.no_grad():
            net.log_std.clamp_(np.log(0.05), np.log(1.0))
        if it % 10 == 0:
            el = np.mean(done_len[-100:]) if done_len else 0
            er = np.mean(done_ret[-100:]) if done_ret else 0
            msg = dict(it=it, steps=steps, fps=int(steps / (time.time() - t0)), ep_len=round(el, 1), ep_ret=round(er, 2),
                       rew=round(R.mean().item(), 3), kl=round(kl, 4), lr=round(lr, 6),
                       std=round(net.log_std.exp().mean().item(), 3),
                       **{k: round(float(np.mean(v)), 3) for k, v in infos.items()})
            print(json.dumps(msg), flush=True); log.write(json.dumps(msg) + '\n'); log.flush()
        if it % 100 == 0 or it == args.iters - 1:
            torch.save({'model': net.state_dict(), 'it': it, 'steps': steps}, os.path.join(args.out, 'model.pt'))
            if it % 500 == 0:
                torch.save({'model': net.state_dict(), 'it': it}, os.path.join(args.out, f'model_{it}.pt'))
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
