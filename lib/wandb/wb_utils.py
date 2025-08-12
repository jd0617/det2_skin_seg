import wandb

from detectron2.engine import HookBase

def init_wandb(cfg):
    wandb.init(
        project = cfg.WANDB.PROJECT,
        name=cfg.WANDB.NAME,
        config={"seed": cfg.SEED}
    )

class WandbHook(HookBase):
    def __init__(self, log_interval=20, watch_model=False):
        super().__init__()
        self.log_interval = log_interval
        self.watch_model = watch_model
    
    def before_train(self):
        if self.watch_model:
            wandb.watch(
                self.trainer.model,
                log='all',
                log_freq=self.log_interval
            )

    def after_step(self):
        it = self.trainer.iter + 1
        if it % self.log_interval != 0:
            return
        
        storage = self.trainer.storage

        scalars = storage.latest_with_smoothing_hint()

        try:
            scalars['lr'] = storage.history('lr').latest()
        except KeyError:
            pass

        wandb.log({k: float(v) for k, v in scalars.items()}, step=it)
