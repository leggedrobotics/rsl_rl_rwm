import torch
import torch.nn as nn


class MLPBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        
        base_shape = architecture_config["base_shape"]
        layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in base_shape:
            layers.append(nn.Linear(curr_in_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        self.layers = nn.Sequential(*layers).to(self.device)
        self.layers.train()
        
    def forward(self, x_state_batch, x_action_batch):
        x = torch.cat([x_state_batch, x_action_batch], dim=-1).flatten(1, 2)
        x = self.layers(x)
        return x
    
    def reset(self):
        pass


class MLPStateHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        device: str,
        architecture_config: dict = None,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.device = device
        self.state_mean_shape = architecture_config["state_mean_shape"]
        self.state_logstd_shape = architecture_config["state_logstd_shape"]

        state_mean_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.state_mean_shape:
            state_mean_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            state_mean_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        state_mean_layers.append(nn.Linear(self.state_mean_shape[-1], state_dim))
        self.state_mean_layers = nn.Sequential(*state_mean_layers).to(self.device)
        self.state_mean_layers.train()

        if self.state_logstd_shape is not None:
            self.output_std = True
            state_logstd_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in self.state_logstd_shape:
                state_logstd_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                state_logstd_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            state_logstd_layers.append(nn.Linear(self.state_logstd_shape[-1], state_dim))
            self.state_logstd_layers = nn.Sequential(*state_logstd_layers).to(self.device)
            self.state_logstd_layers.train()
        else:
            self.output_std = False

        if self.output_std:
            self.state_min_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * -5.0)
            self.state_log_delta_logstd = nn.Parameter(torch.ones(1, state_dim, device=self.device) * 0.0)

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
            x_state_batch = x_state_batch.flatten(0, 1).unsqueeze(1)
        else:
            sequence_len = 0
        state_mean = self.state_mean_layers(x) + x_state_batch[:, -1]
        state_logstd = self.state_logstd_layers(x) if self.output_std else -torch.inf * torch.ones(x.shape[0], self.state_dim, device=self.device)
        if self.output_std:
            self.state_max_logstd = self.state_min_logstd + torch.exp(self.state_log_delta_logstd)
            state_logstd = self.state_max_logstd - nn.functional.softplus(self.state_max_logstd - state_logstd)
            state_logstd = self.state_min_logstd + nn.functional.softplus(state_logstd - self.state_min_logstd)
        if sequence_len > 0:
            state_mean = state_mean.view(-1, sequence_len, self.state_dim)
            state_logstd = state_logstd.view(-1, sequence_len, self.state_dim)
        return state_mean, torch.exp(state_logstd)

    def reset(self):
        pass


class MLPAuxiliaryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        architecture_config: dict = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device

        if extension_dim > 0:
            extension_shape = architecture_config["extension_shape"]
            extension_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in extension_shape:
                extension_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                extension_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            extension_layers.append(nn.Linear(extension_shape[-1], extension_dim))
            self.extension_layers = nn.Sequential(*extension_layers).to(self.device)
            self.extension_layers.train()

        if contact_dim > 0:
            contact_shape = architecture_config["contact_shape"]
            contact_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in contact_shape:
                contact_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                contact_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            contact_layers.append(nn.Linear(contact_shape[-1], contact_dim))
            self.contact_layers = nn.Sequential(*contact_layers).to(self.device)
            self.contact_layers.train()
        
        if termination_dim > 0:
            termination_shape = architecture_config["termination_shape"]
            termination_layers = []
            curr_in_dim = self.input_dim
            for hidden_dim in termination_shape:
                termination_layers.append(nn.Linear(curr_in_dim, hidden_dim))
                termination_layers.append(nn.ReLU())
                curr_in_dim = hidden_dim
            termination_layers.append(nn.Linear(termination_shape[-1], termination_dim))
            self.termination_layers = nn.Sequential(*termination_layers).to(self.device)
            self.termination_layers.train()

    def forward(self, x, x_state_batch):
        if x.dim() == 3:
            sequence_len = x.shape[1]
            x = x.flatten(0, 1)
        else:
            sequence_len = 0
        
        extension_pred = self.extension_layers(x) if self.extension_dim > 0 else None
        contact_logits = self.contact_layers(x) if self.contact_dim > 0 else None
        termination_logits = self.termination_layers(x) if self.termination_dim > 0 else None
        
        if sequence_len > 0:
            extension_pred = extension_pred.view(-1, sequence_len, self.extension_dim) if self.extension_dim > 0 else None
            contact_logits = contact_logits.view(-1, sequence_len, self.contact_dim) if self.contact_dim > 0 else None
            termination_logits = termination_logits.view(-1, sequence_len, self.termination_dim) if self.termination_dim > 0 else None
        
        return extension_pred, contact_logits, termination_logits

    def reset(self):
        pass
