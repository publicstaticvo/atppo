from torch.nn.parallel import DistributedDataParallel


class DDP(DistributedDataParallel):

    def __init__(self, *args, **kwargs):
        super(DDP, self).__init__(*args, **kwargs)
        
    def __getattr__(self, name):
        try:
            return super(DDP, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
