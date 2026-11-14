class NerOutput(object):
    """Container chuẩn cho đầu ra của mô hình NER."""
    def __init__(self, loss=None, logits=None, tags=None, loss_supcon=None, loss_ce=None):
        self.loss = loss
        self.logits = logits
        self.tags = tags
        self.loss_supcon = loss_supcon
        self.loss_ce = loss_ce

