import torch


class Plotter:


    def plot_pca(self, ax, data_list, legend_list=None):
        ax.cla()
        data_all = torch.cat(data_list, dim=0).cpu()
        split_ids = [len(data) for data in data_list]
        _, _, V = torch.pca_lowrank(data_all)
        data_all_pca = torch.matmul(data_all, V[:, :2])
        data_pca = torch.split(data_all_pca, split_ids)
        for i, data in enumerate(data_pca):
            ax.scatter(data[:, 0], data[:, 1], alpha=0.4, label=legend_list[i] if legend_list is not None else None)
        ax.legend()
        ax.set_axis_off()
        
    def plot_trajectories(self, axes, start_step, state_traj, action_traj, extension_traj, contact_traj, termination_traj, state_idx_dict, prediction=False):
        state_traj = state_traj.cpu()
        
        if axes.ndim == 1:
            axes = axes[:, None]
        
        for i, state in enumerate(state_traj):
            for j, (state_label, state_idx) in enumerate(state_idx_dict.items()):
                ax = axes[j, i]
                if prediction:
                    ax.set_prop_cycle(None)
                    ax.plot(state[:, state_idx], alpha=0.5, linestyle='--')
                    ax.axvline(start_step, color="tab:grey", linewidth=1.0)
                else:
                    ax.cla()
                    ax.plot(state[:, state_idx], alpha=0.2)
                    ax.set_ylim(state[:, state_idx].min() * 1.1, state[:, state_idx].max() * 1.1)
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_ylabel(r'{}'.format(state_label))
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.margins(x=0.0)
                ax.grid(True, linestyle="--", alpha=0.5)

            last_ax = ax  # Keep track of the last ax used

            if action_traj is not None:
                action = action_traj[i].cpu()
                ax = axes[j + 1, i]
                if prediction:
                    ax.set_prop_cycle(None)
                    ax.plot(action, alpha=0.5, linestyle='--')
                    ax.axvline(start_step, color="tab:grey", linewidth=1.0)
                else:
                    ax.cla()
                    ax.plot(action, alpha=0.2)
                    ax.set_ylim(action.min(), action.max())
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_ylabel(r'{}'.format("$a$\n$[rad]$"))
                ax.margins(x=0.0)
                ax.grid(True, linestyle="--", alpha=0.5)
                last_ax = ax

            if extension_traj is not None:
                extension = extension_traj[i].cpu()
                ax = axes[j + 2, i]
                if prediction:
                    ax.set_prop_cycle(None)
                    ax.plot(extension, alpha=0.5, linestyle='--')
                    ax.axvline(start_step, color="tab:grey", linewidth=1.0)
                else:
                    ax.cla()
                    ax.plot(extension, alpha=0.2)
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_ylabel(r'{}'.format("$ext$\n$[1]$"))
                ax.margins(x=0.0)
                ax.grid(True, linestyle="--", alpha=0.5)
                last_ax = ax

            if contact_traj is not None:
                contact = contact_traj[i].cpu()
                ax = axes[j + 3, i]
                if prediction:
                    ax.set_prop_cycle(None)
                    ax.plot(contact, alpha=0.5, linestyle='--')
                    ax.axvline(start_step, color="tab:grey", linewidth=1.0)
                else:
                    ax.cla()
                    ax.plot(contact, alpha=0.2)
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_ylabel(r'{}'.format("$c$\n$[1]$"))
                ax.margins(x=0.0)
                ax.grid(True, linestyle="--", alpha=0.5)
                last_ax = ax

            if termination_traj is not None:
                termination = termination_traj[i].cpu()
                ax = axes[j + 4, i]
                if prediction:
                    ax.set_prop_cycle(None)
                    ax.plot(termination, alpha=0.5, linestyle='--')
                    ax.axvline(start_step, color="tab:grey", linewidth=1.0)
                else:
                    ax.cla()
                    ax.plot(termination, alpha=0.2)
                ax.set_ylabel(r'{}'.format("$term$\n$[1]$"))
                ax.margins(x=0.0)
                ax.grid(True, linestyle="--", alpha=0.5)
                last_ax = ax

            if start_step is not None:
                last_ax.set_xticks(list(ax.get_xticks()) + [start_step])
            else:
                last_ax.set_xticks(list(ax.get_xticks()))
            last_ax.set_xticklabels([str(int(val)) for val in list(ax.get_xticks())])
            last_ax.set_xlabel(r'$t$')  # Set xlabel on the last ax used
