import time
import logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        fmtstr = '{name}-{avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__) 

    def __str__(self):
        fmtstr = '{name}-{val' + self.fmt + '}({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class MaxMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.max = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if val > self.max:
            self.max = val

    def __repr__(self):
        fmtstr = '{name}-{max' + self.fmt + '}'
        return fmtstr.format(**self.__dict__) 

    def __str__(self):
        fmtstr = '{name}-{val' + self.fmt + '}({max' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class CumMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def __repr__(self):
        fmtstr = '{name}-{sum' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)        

    def __str__(self):
        fmtstr = '{name}-{val' + self.fmt + '}({sum' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", length=10):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.num_batches = num_batches
        self.length = length
        self.interval = num_batches // length
        self.atd = time.time()

    def display(self, batch, log_level=logging.DEBUG, reduce=False):
        # entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # entries += [str(meter) for meter in self.meters]
        # print('\t'.join(entries))
        collapse_time = time.time() - self.atd
        eta = collapse_time / (batch+1) * (self.num_batches - batch -1)

        count = (batch + self.interval - 1) / self.num_batches * self.length
        count = int(count)
        entries = [self.prefix + '['+'#'*count + ' '*(self.length - count)+']']
        entries += ['{:.0f}s'.format(collapse_time)]
        if reduce:
            entries += [repr(meter) for meter in self.meters]
        else:
            entries += ['eta- {:.0f}s'.format(eta)]
            entries += [str(meter) for meter in self.meters]
        print('  '.join(entries), end='\r', flush=True)
        logging.log(log_level, '  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

