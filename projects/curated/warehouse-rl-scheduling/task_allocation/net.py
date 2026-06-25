import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_mean(x, mask, dim=1, eps=1e-8):
    w = mask.float().unsqueeze(-1)
    s = (x * w).sum(dim=dim)
    c = w.sum(dim=dim).clamp(min=eps)
    return s / c


class ActorNet(nn.Module):
    def __init__(self, agent_dim=6, task_dim=5, hidden_dim=128):
        super().__init__()

        self.agent_encoder = MLP(agent_dim, [hidden_dim], hidden_dim)

        self.task_encoder = MLP(task_dim + 1, [hidden_dim], hidden_dim)

        self.context_fuser = MLP(
            in_dim=hidden_dim * 2,
            hidden_dims=[hidden_dim],
            out_dim=hidden_dim
        )

        self.policy_head = MLP(
            in_dim=hidden_dim * 2,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=1
        )

    def forward(self, agents_info, task_info, mask_info):
        B, _, _ = agents_info.shape
        _, N_tasks, _ = task_info.shape

        agent_emb = self.agent_encoder(agents_info)      # (B, 10, H)
        agent_ctx = agent_emb.mean(dim=1)                # (B, H)

        available = (~mask_info.bool()).float().unsqueeze(-1)   # (B, 17, 1)
        task_input = torch.cat([task_info, available], dim=-1)  # (B, 17, 6)

        task_emb = self.task_encoder(task_input)         # (B, 17, H)

        valid_task_ctx = masked_mean(task_emb, ~mask_info.bool(), dim=1)   # (B, H)

        global_ctx = self.context_fuser(
            torch.cat([agent_ctx, valid_task_ctx], dim=-1)
        )                                                # (B, H)

        global_expand = global_ctx.unsqueeze(1).expand(-1, N_tasks, -1)    # (B, 17, H)
        fused = torch.cat([task_emb, global_expand], dim=-1)                # (B, 17, 2H)

        logits = self.policy_head(fused).squeeze(-1)                        # (B, 17)

        logits = logits.masked_fill(mask_info.bool(), -1e9)

        return logits


class CriticNet(nn.Module):
    def __init__(self, agent_dim=6, task_dim=5, hidden_dim=128):
        super().__init__()

        self.agent_encoder = MLP(agent_dim, [hidden_dim], hidden_dim)
        self.task_encoder = MLP(task_dim + 1, [hidden_dim], hidden_dim)

        self.value_head = MLP(
            in_dim=hidden_dim * 3 + 2,
            hidden_dims=[hidden_dim, hidden_dim],
            out_dim=1
        )

    def forward(self, agents_info, task_info, mask_info):
        agent_emb = self.agent_encoder(agents_info)      # (B, 10, H)
        agent_ctx_mean = agent_emb.mean(dim=1)           # (B, H)
        agent_ctx_max = agent_emb.max(dim=1).values      # (B, H)

        available = (~mask_info.bool()).float().unsqueeze(-1)   # (B, 17, 1)
        task_input = torch.cat([task_info, available], dim=-1)  # (B, 17, 6)
        task_emb = self.task_encoder(task_input)                 # (B, 17, H)

        valid_mask = ~mask_info.bool()
        task_ctx = masked_mean(task_emb, valid_mask, dim=1)      # (B, H)

        num_available = available.sum(dim=1)                     # (B, 1)
        frac_available = num_available / task_info.size(1)       # (B, 1)

        state_feat = torch.cat(
            [agent_ctx_mean, agent_ctx_max, task_ctx, num_available, frac_available],
            dim=-1
        )                                                        # (B, 3H+2)

        value = self.value_head(state_feat)                      # (B, 1)
        return value


class ActorCritic(nn.Module):
    def __init__(self, agent_dim=6, task_dim=5, hidden_dim=128):
        super().__init__()
        self.actor = ActorNet(agent_dim, task_dim, hidden_dim)
        self.critic = CriticNet(agent_dim, task_dim, hidden_dim)

    def forward(self, agents_info, task_info, mask_info):
        logits = self.actor(agents_info, task_info, mask_info)
        value = self.critic(agents_info, task_info, mask_info)
        return logits, value

    def act(self, agents_info, task_info, mask_info, deterministic=False):
        logits = self.actor(agents_info, task_info, mask_info)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(agents_info, task_info, mask_info).squeeze(-1)

        return action, log_prob, entropy, value