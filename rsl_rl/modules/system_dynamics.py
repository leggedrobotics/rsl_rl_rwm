import torch
import torch.nn as nn
from rsl_rl.modules.architectures import MLPBase, RNNBase, MLPStateHead, MLPAuxiliaryHead

class SystemDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        ensemble_size: int = 1,
        history_horizon: int = 1,
        architecture_config: dict = None,
        freeze_auxiliary: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config
        self.freeze_auxiliary = freeze_auxiliary
        
        self._init_networks()

    def _init_networks(self):
        self.state_base = self._create_base()
        self.state_heads = nn.ModuleList([
            MLPStateHead(
                self.base_output_dim,
                self.state_dim,
                self.device,
                self.architecture_config
            ).to(self.device) for _ in range(self.ensemble_size)
        ])

        self.auxiliary_base = self._create_base()
        self.auxiliary_heads = nn.ModuleList([
            MLPAuxiliaryHead(
                self.base_output_dim,
                self.extension_dim,
                self.contact_dim,
                self.termination_dim,
                self.device,
                self.architecture_config
            ).to(self.device) for _ in range(self.ensemble_size)
        ])

        if self.freeze_auxiliary:
            for param in self.auxiliary_base.parameters():
                param.requires_grad = False
            for head in self.auxiliary_heads:
                for param in head.parameters():
                    param.requires_grad = False

    def _create_base(self):
        if self.architecture_config["type"] == "mlp":
            input_dim = self.history_horizon * (self.state_dim + self.action_dim)
            self.base_output_dim = self.architecture_config["base_shape"][-1]
            self.prediction_type = "single"
            return MLPBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config,
            )
        elif self.architecture_config["type"] == "rnn":
            input_dim = self.state_dim + self.action_dim
            self.base_output_dim = self.architecture_config["rnn_hidden_size"]
            self.prediction_type = "single"
            return RNNBase(
                input_dim=input_dim,
                device=self.device,
                architecture_config=self.architecture_config
            )
        else:
            raise ValueError("Invalid architecture type.")

    def forward(self, x_state_batch, x_action_batch, model_ids=None):
        state_means, state_stds, extensions, contacts, terminations = [], [], [], [], []
        state_base_output = self.state_base(x_state_batch, x_action_batch)
        
        for head in self.state_heads:
            state_mean, state_std = head(state_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                state_mean = state_mean[:, -1]
                state_std = state_std[:, -1]
            state_means.append(state_mean.unsqueeze(0))
            state_stds.append(state_std.unsqueeze(0))

        auxiliary_base_output = self.auxiliary_base(x_state_batch, x_action_batch)
        for head in self.auxiliary_heads:
            extension, contact, termination = head(auxiliary_base_output, x_state_batch)
            if self.prediction_type == "sequence":
                extension = extension[:, -1] if extension is not None else None
                contact = contact[:, -1] if contact is not None else None
                termination = termination[:, -1] if termination is not None else None
            extensions.append(extension.unsqueeze(0) if extension is not None else None)
            contacts.append(contact.unsqueeze(0) if contact is not None else None)
            terminations.append(termination.unsqueeze(0) if termination is not None else None)

        state_means = torch.cat(state_means, dim=0)
        state_stds = torch.cat(state_stds, dim=0)
        extensions = torch.cat(extensions, dim=0) if self.extension_dim > 0 else None
        contacts = torch.cat(contacts, dim=0) if self.contact_dim > 0 else None
        terminations = torch.cat(terminations, dim=0) if self.termination_dim > 0 else None
        
        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
            output_extensions = extensions.mean(dim=0) if extensions is not None else None
            output_contacts = contacts.mean(dim=0) if contacts is not None else None
            output_terminations = terminations.mean(dim=0) if terminations is not None else None
        else:
            output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.state_dim)).squeeze(0)
            output_extensions = torch.gather(extensions, 0, model_ids.repeat(1, 1, self.extension_dim)).squeeze(0) if extensions is not None else None
            output_contacts = torch.gather(contacts, 0, model_ids.repeat(1, 1, self.contact_dim)).squeeze(0) if contacts is not None else None
            output_terminations = torch.gather(terminations, 0, model_ids.repeat(1, 1, self.termination_dim)).squeeze(0) if terminations is not None else None
        
        aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
        epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 else torch.zeros(output_state_means.shape[0], device=self.device)
        return output_state_means, aleatoric_uncertainty, epistemic_uncertainty, output_extensions, output_contacts, output_terminations

    def compute_loss(self, state_batch, action_batch, extension_batch, contact_batch, termination_batch, bootstrap=False):
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        extension_losses = []
        contact_losses = []
        termination_losses = []
        
        for i in range(self.ensemble_size):
            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=self.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=self.device)
            
            state_loss, sequence_loss, bound_loss, kl_loss = self.compute_state_loss(
                self.state_heads[i], state_batch[ids], action_batch[ids]
            )
            
            if self.auxiliary_heads is not None:
                extension_loss, contact_loss, termination_loss = self.compute_auxiliary_loss(
                    self.auxiliary_heads[i],
                    state_batch[ids],
                    action_batch[ids],
                    extension_batch[ids] if extension_batch is not None else None,
                    contact_batch[ids] if contact_batch is not None else None,
                    termination_batch[ids] if termination_batch is not None else None
                )
            else:
                extension_loss = torch.tensor(0.0, device=self.device)
                contact_loss = torch.tensor(0.0, device=self.device)
                termination_loss = torch.tensor(0.0, device=self.device)
            
            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))
        
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss, extension_loss, contact_loss, termination_loss

    def compute_state_loss(self, head, state_batch, action_batch):
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        x_state_batch = state_batch[:, :self.history_horizon]
        state_losses = []
        sequence_losses = []
        bound_losses = []
        kl_losses = []
        
        for i in range(forecast_horizon):
            if self.prediction_type == "single":
                state_target = state_batch[:, self.history_horizon + i]
            elif self.prediction_type == "sequence":
                state_target = state_batch[:, i + 1:self.history_horizon + i + 1]
            else:
                raise ValueError("Invalid state prediction type.")
            
            if self.architecture_config["type"] in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
                if self.prediction_type == "sequence":
                    state_target = state_target[:, [-1]]
            else:
                x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
            
            state_mean_pred, state_std_pred = head.forward(self.state_base.forward(x_state_batch, x_action_batch), x_state_batch)
            state_loss, sequence_loss = self.compute_regression_loss(state_mean_pred, state_std_pred, state_target)
            bound_loss = self.compute_bound_loss(head) if head.output_std else torch.tensor(0.0, device=self.device)
            kl_loss = self.state_base.kl_loss if self.architecture_config["type"] == "rssm" else torch.tensor(0.0, device=self.device)
            
            state_losses.append(state_loss.unsqueeze(0))
            sequence_losses.append(sequence_loss.unsqueeze(0))
            bound_losses.append(bound_loss.unsqueeze(0))
            kl_losses.append(kl_loss.unsqueeze(0))
            
            if self.prediction_type == "sequence":
                state_mean_pred = state_mean_pred[:, -1]
                state_std_pred = state_std_pred[:, -1]
            
            if self.architecture_config["type"] in ["rnn", "rssm"]:
                x_state_batch = (torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred).unsqueeze(1) if head.output_std else state_mean_pred.unsqueeze(1)
            else:
                x_state_batch = torch.cat(
                    [
                        x_state_batch[:, 1:].clone(),
                        (torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred).unsqueeze(1) if head.output_std else state_mean_pred.unsqueeze(1),
                    ],
                    dim=1
                )
        
        state_loss = torch.mean(torch.cat(state_losses, dim=0), dim=0)
        sequence_loss = torch.mean(torch.cat(sequence_losses, dim=0), dim=0)
        bound_loss = torch.mean(torch.cat(bound_losses, dim=0), dim=0)
        kl_loss = torch.mean(torch.cat(kl_losses, dim=0), dim=0)
        return state_loss, sequence_loss, bound_loss, kl_loss

    def compute_auxiliary_loss(self, head, state_batch, action_batch, extension_batch, contact_batch, termination_batch):
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        x_state_batch = state_batch[:, :self.history_horizon]
        extension_losses = []
        contact_losses = []
        termination_losses = []
        
        for i in range(forecast_horizon):
            extension_target = extension_batch[:, self.history_horizon + i] if extension_batch is not None else None
            contact_target = contact_batch[:, self.history_horizon + i] if contact_batch is not None else None
            termination_target = termination_batch[:, self.history_horizon + i] if termination_batch is not None else None
            
            if self.architecture_config["type"] in ["rnn", "rssm"] and i > 0:
                x_action_batch = action_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
            else:
                x_action_batch = action_batch[:, i + 1:self.history_horizon + i + 1]
            
            extension_pred, contact_pred, termination_pred = head.forward(self.auxiliary_base.forward(x_state_batch, x_action_batch), x_state_batch)
            
            extension_loss = self.compute_extension_loss(extension_pred, extension_target) if self.extension_dim > 0 else torch.tensor(0.0, device=self.device)
            contact_loss = self.compute_contact_loss(contact_pred, contact_target) if self.contact_dim > 0 else torch.tensor(0.0, device=self.device)
            termination_loss = self.compute_termination_loss(termination_pred, termination_target) if self.termination_dim > 0 else torch.tensor(0.0, device=self.device)
            
            extension_losses.append(extension_loss.unsqueeze(0))
            contact_losses.append(contact_loss.unsqueeze(0))
            termination_losses.append(termination_loss.unsqueeze(0))
            
            if self.architecture_config["type"] in ["rnn", "rssm"]:
                x_state_batch = state_batch[:, self.history_horizon + i:self.history_horizon + i + 1]
            else:
                x_state_batch = torch.cat([x_state_batch[:, 1:].clone(), state_batch[:, self.history_horizon + i:self.history_horizon + i + 1]], dim=1)
        
        extension_loss = torch.mean(torch.cat(extension_losses, dim=0), dim=0)
        contact_loss = torch.mean(torch.cat(contact_losses, dim=0), dim=0)
        termination_loss = torch.mean(torch.cat(termination_losses, dim=0), dim=0)
        return extension_loss, contact_loss, termination_loss

    def compute_regression_loss(self, state_mean_pred, state_std_pred, state_target, loss_type="mse"):
        if loss_type == "mse":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_pred_seq = torch.randn_like(state_mean_pred_seq, device=self.device) * state_std_pred_seq + state_mean_pred_seq
                state_pred_seq = state_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = torch.sum(torch.square(state_pred_seq - state_target_seq), dim=1).mean(dim=0)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_pred = torch.randn_like(state_mean_pred, device=self.device) * state_std_pred + state_mean_pred
            state_loss = torch.sum(torch.square(state_pred - state_target), dim=1).mean(dim=0)
            return state_loss, sequence_loss
        elif loss_type == "gaussian_nll":
            if self.prediction_type == "sequence":
                state_mean_pred_seq, state_mean_pred = state_mean_pred[:, :-1], state_mean_pred[:, -1]
                state_std_pred_seq, state_std_pred = state_std_pred[:, :-1], state_std_pred[:, -1]
                state_target_seq, state_target = state_target[:, :-1], state_target[:, -1]
                state_mean_pred_seq = state_mean_pred_seq.flatten(0, 1)
                state_std_pred_seq = state_std_pred_seq.flatten(0, 1)
                state_target_seq = state_target_seq.flatten(0, 1)
                sequence_loss = nn.GaussianNLLLoss()(state_mean_pred_seq, state_target_seq, state_std_pred_seq ** 2)
            else:
                sequence_loss = torch.tensor(0.0, device=self.device)
            state_loss = nn.GaussianNLLLoss()(state_mean_pred, state_target, state_std_pred ** 2)
            return state_loss, sequence_loss
        else:
            raise ValueError("Invalid loss type.")
        
    def compute_bound_loss(self, head):
        return torch.mean(head.state_max_logstd) - torch.mean(head.state_min_logstd)

    def compute_extension_loss(self, extension_pred, extension_target):
        if extension_pred is None or extension_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            extension_pred = extension_pred[:, -1]
        return nn.MSELoss()(extension_pred, extension_target)
    
    def compute_contact_loss(self, contact_pred, contact_target):
        if contact_pred is None or contact_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            contact_pred = contact_pred[:, -1]
        return nn.BCEWithLogitsLoss()(contact_pred, contact_target)
    
    def compute_termination_loss(self, termination_pred, termination_target):
        if termination_pred is None or termination_target is None:
            return torch.tensor(0.0, device=self.device)
        if self.prediction_type == "sequence":
            termination_pred = termination_pred[:, -1]
        return nn.BCEWithLogitsLoss()(termination_pred, termination_target)

    def reset(self):
        self.state_base.reset()
        for head in self.state_heads:
            head.reset()
        if self.auxiliary_base is not None:
            self.auxiliary_base.reset()
            for head in self.auxiliary_heads:
                head.reset()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_auxiliary:
            self.auxiliary_base.eval()
            for head in self.auxiliary_heads:
                head.eval()
        return self