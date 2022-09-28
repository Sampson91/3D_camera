from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from timm.scheduler.scheduler import Scheduler


class PolynomialLR(_LRScheduler):
    def __init__(
            self,
            optimizer,
            step_size,
            iteration_warmup,
            iteration_max,
            power,
            min_learning_rate=0,
            last_epoch=-1,):
        self.step_size = step_size
        self.iteration_warmup = int(iteration_warmup)
        self.iteration_max = int(iteration_max)
        self.power = power
        self.min_learning_rate = min_learning_rate
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, learning_rate):
        iteration_current = float(self.last_epoch)
        if iteration_current < self.iteration_warmup:
            coefficient = iteration_current / self.iteration_warmup
            coefficient *= (
                                       1 - self.iteration_warmup / self.iteration_max) ** self.power
        else:
            coefficient = (
                                      1 - iteration_current / self.iteration_max) ** self.power
        return (
                           learning_rate - self.min_learning_rate) * coefficient + self.min_learning_rate

    def get_lr(self):
        # get_lr super to _LRScheduler
        if (
                (self.last_epoch == 0)
                or (self.last_epoch % self.step_size != 0)
                or (self.last_epoch > self.iteration_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
            # lr optimizer 内部的
        return [self.polynomial_decay(learning_rate) for learning_rate in
                self.base_lrs]

    def step_update(self, num_updates):
        self.step()
