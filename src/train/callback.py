import torch
from collections import defaultdict
from transformers import TrainerState, TrainerControl, TrainingArguments
from transformers.integrations import WandbCallback

class FisherCallback(WandbCallback):
    def __init__(
        self,
        keys: list[str],
        batch_size: int,
    ):
        super().__init__()
        self.keys = sorted(keys, key=len, reverse=True)
        self.num_samples = None
        self.batch_size = batch_size

    def _assign_group(self, param_name: str) -> str:
        for key in self.keys:
            if param_name.startswith(key): return key
        return "__other__"

    def _compute_fisher(
        self,
        args: TrainingArguments,
        model,
        eval_dataloader,
    ) -> dict[str, float]:
        model.eval()

        fisher_accum: dict[str, torch.Tensor] = {}
        n_samples = 0

        for batch in eval_dataloader:
            if self.num_samples is not None and n_samples >= self.num_samples:
                break

            batch = {
                k: v.to(args.device)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            batch_size = next(iter(batch.values())).shape[0]

            model.zero_grad()
            with torch.enable_grad():
                outputs = model(**batch)
                outputs.loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    sq = param.grad.detach().pow(2).mul_(batch_size)
                    fisher_accum[name] = fisher_accum.get(name, torch.zeros_like(sq)).add_(sq)

            n_samples += batch_size

        model.zero_grad()

        if n_samples == 0: return {}

        group_values: dict[str, list[float]] = defaultdict(list)
        for name, accum in fisher_accum.items():
            fisher_val = accum.div_(n_samples).sum().item()
            group_values[self._assign_group(name)].append(fisher_val)

        return {key: sum(vals) for key, vals in group_values.items()}

    def _log_to_wandb(self, group_fisher: dict[str, float], step: int):
        data = {f"fisher/{key}": val for key, val in group_fisher.items()}
        self._wandb.log(data, step=step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        eval_dataloader=None,
        **kwargs,
    ):
        out = super().on_evaluate(args, state, control, **kwargs)
        group_fisher = self._compute_fisher(args, model, eval_dataloader)
        self._log_to_wandb(group_fisher, step=state.global_step)
        self._print_summary(group_fisher, state.global_step)
        return out

    def _print_summary(self, group_fisher: dict, step: int):
        label = f"step {step}"
        print(f"\n[FisherCallback] {label} — total empirical Fisher per group")
        print(f"  {'Group':<40} {'Fisher (sum)':>15}")
        print(f"  {'-'*40} {'-'*15}")
        for key in self.keys + (["__other__"] if "__other__" in group_fisher else []):
            if key in group_fisher: print(f"  {key:<40} {group_fisher[key]:>15.6e}")
        print()