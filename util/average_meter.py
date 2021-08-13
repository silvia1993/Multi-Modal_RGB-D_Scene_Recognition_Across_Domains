class AverageMeter(object):
    
    """Computes and stores the average and current value"""
    
    def __init__(self):
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
        self.avg = round(self.sum / self.count, 3)

    def __str__(self): #toString method equivalent
        return ("##################################\n"
            "Current val : " + str(self.val) + "\n"
            "Current sum : " + str(self.sum) + "\n"
            "Current count : " + str(self.count) + "\n"
            "Current avg : " + str(self.avg) + "\n")