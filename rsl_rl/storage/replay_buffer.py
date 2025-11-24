import torch
import numpy as np


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, dim, buffer_size, device):
        """Initialize a ReplayBuffer object."""
        self.dim = dim
        self.buffer_size = buffer_size
        self.device = device

        self.replay_buf = None
        self.num_envs = None
        self.step = 0
        self.num_transitions = 0

    def _initialize_buffer(self, num_envs):
        """Initialize the replay buffer dynamically."""
        self.num_envs = num_envs
        if isinstance(self.dim, list):
            self.replay_buf = [
                torch.zeros(num_envs, self.buffer_size, d, device=self.device)
                if d > 0 else None for d in self.dim
            ]
        else:
            self.replay_buf = torch.zeros(num_envs, self.buffer_size, self.dim, device=self.device)

    def insert(self, input_buf):
        """Add new states to memory in a circular manner."""
        if self.replay_buf is None:
            self._initialize_buffer(input_buf[0].shape[0] if isinstance(input_buf, list) else input_buf.shape[0])

        def _insert_into_buffer(r_buf, i_buf):
            num_inputs = i_buf.shape[1]
            end_idx = self.step + num_inputs
            if end_idx > self.buffer_size:
                r_buf[:, self.step:self.buffer_size] = i_buf[:, :self.buffer_size - self.step]
                r_buf[:, :end_idx - self.buffer_size] = i_buf[:, self.buffer_size - self.step:]
            else:
                r_buf[:, self.step:end_idx] = i_buf
            return num_inputs

        num_inputs = 0
        if isinstance(self.replay_buf, list):
            for r_buf, i_buf in zip(self.replay_buf, input_buf):
                if r_buf is not None:
                    num_inputs = _insert_into_buffer(r_buf, i_buf)
        else:
            num_inputs = _insert_into_buffer(self.replay_buf, input_buf)

        self.num_transitions = min(self.buffer_size, self.num_transitions + num_inputs)
        self.step = (self.step + num_inputs) % self.buffer_size

    def mini_batch_generator(self, sequence_length, num_mini_batch, mini_batch_size):
        """Yield mini-batches for training."""
        assert self.replay_buf is not None, "Replay buffer is not initialized."

        def _pad_and_get_replay_buf():
            padding_size = sequence_length - self.num_transitions
            if isinstance(self.replay_buf, list):
                return [torch.cat([torch.zeros(buf.shape[0], padding_size, buf.shape[-1], device=self.device),
                                   buf[:, :self.num_transitions]], dim=1) if buf is not None else None
                        for buf in self.replay_buf]
            else:
                return torch.cat([torch.zeros(self.replay_buf.shape[0], padding_size, self.replay_buf.shape[-1],
                                              device=self.device),
                                  self.replay_buf[:, :self.num_transitions]], dim=1)

        replay_buf = _pad_and_get_replay_buf() if self.num_transitions < sequence_length else self.replay_buf
        reset_data = replay_buf[-1] if isinstance(replay_buf, list) else None

        valid_indices = None
        if reset_data is not None:
            valid_indices = self._generate_valid_indices(reset_data, sequence_length)

        for _ in range(num_mini_batch):
            yield self._generate_batch(replay_buf, valid_indices, sequence_length, mini_batch_size)

    def _generate_valid_indices(self, reset_data, sequence_length):
        """Generate valid start indices for sequences."""
        reset_flags = reset_data[:, :max(self.num_transitions, sequence_length)].to(torch.bool)
        valid_mask = ~reset_flags.unfold(1, sequence_length, 1).squeeze(-2)[:, :, :-1].any(dim=2)
        env_indices, start_indices = torch.where(valid_mask)
        return env_indices, start_indices

    def _generate_batch(self, replay_buf, valid_indices, sequence_length, mini_batch_size):
        """Generate a mini-batch by sampling valid sequences."""
        if valid_indices is None:
            max_start_idx = max(self.num_transitions - sequence_length, 0) + 1
            sampled_envs = torch.tensor(np.random.choice(self.num_envs, size=mini_batch_size), device=self.device)
            sampled_starts = torch.tensor(np.random.choice(max_start_idx, size=mini_batch_size), device=self.device)
            offsets = torch.arange(sequence_length, device=self.device)
            if isinstance(replay_buf, list):
                return [buf[sampled_envs[:, None], sampled_starts[:, None] + offsets] if buf is not None else None for buf in replay_buf]
            else:
                return replay_buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]

        env_indices, start_indices = valid_indices
        sampled_idxs = torch.tensor(np.random.choice(len(env_indices), size=mini_batch_size), device=self.device)
        sampled_envs = env_indices[sampled_idxs]
        sampled_starts = start_indices[sampled_idxs]
        offsets = torch.arange(sequence_length, device=self.device)

        if isinstance(replay_buf, list):
            return [buf[sampled_envs[:, None], sampled_starts[:, None] + offsets] if buf is not None else None
                    for buf in replay_buf]
        else:
            return replay_buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]
