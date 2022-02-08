import torch
from torchmetrics import Metric
from torchmetrics import Recall

class EMR_mt(Metric):
    def __init__(self, use_logits=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.use_logits = use_logits
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        #assert preds.shape == target.shape
        #print(preds.shape, targets.shape)
        if self.use_logits:
            y_hat = (preds > 0.0)
        else:
            y_hat = (preds > 0.5)

        z = (y_hat == targets)
        #print(z)
        z = torch.all(z, dim=1)
        #print(z)
        z = z.sum()
        #print(z)
        self.correct += z
        self.total += targets.shape[0]
        #print("+"*50)

    def compute(self):
        return self.correct.float() / self.total


class Recalls_mt(Recall):
    def __init__(self, average='none', num_classes=6):
        """ Initalisieren über Eltern-Klasse """
        super().__init__(average=average, num_classes=num_classes)    

if __name__ == '__main__':
    myemr = EMR_mt(use_logits=False)
    myrecalls = Recalls_mt()

    # data
    preds0  = torch.tensor([[.9, 0.1, 0.9, 0.1, 0.9, 0.1], 
                           [.8, 0.2, 0.9, 0.2, 0.9, 0.2], 
                           [.7, 0.9, 0.2 , 0.2, 0.2 , 0.2]])
    preds1 = torch.tensor([[.0, 0.1, 0.9, 0.1, 0.9, 0.1], 
                       [.8, 0.2, 0.9, 0.2, 0.9, 0.2], 
                       [.7, 0.9, 0.2 , 0.9, 0.2 , 0.9]])
    target = torch.tensor([[1, 0, 1, 0, 0, 1], 
                            [1, 1, 0, 0, 1, 0], 
                             [1, 1, 0, 1, 0, 1]])
    # batch 0
    myemr(preds0, target), myrecalls(preds0, target)
    print(myemr.compute(), myrecalls.compute())

    # batch 1
    myemr(preds1, target), myrecalls(preds1, target)    
    print(myemr.compute(), myrecalls.compute())

    # Reset at end of epoch
    myemr.reset(), myrecalls.reset()
    print(myemr, myrecalls)