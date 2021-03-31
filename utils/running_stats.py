import torch


# https://stackoverflow.com/a/17637351/1122681
class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = None
        self.new_m = None
        self.old_s = None
        self.new_s = None
        self.sum = None

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.sum = torch.zeros_like(x)
            self.sum += x
            self.old_m = self.new_m = x
            self.old_s = torch.zeros_like(x)
            self.new_m = torch.zeros_like(x)
            self.new_s = torch.zeros_like(x)

        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)
            self.sum += x
            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else torch.zeros_like(self.new_m)

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def std(self):
        return torch.sqrt(self.variance())

    def std_err(self):
        return self.std() / (self.n ** 0.5)

    def get_summary(self):
        return {
            'n': self.n,
            'mean': self.mean(),
            'std': self.std(),
            'full_mean': self.sum / self.n
        }


if __name__ == "__main__":
    stats = RunningStats()
    for i in range(1, 10):
        stats.push(torch.randn((3, 4)))
    print(stats.mean())
    print(stats.std())
