import torch
import tqdm

class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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

class Engine:
    @staticmethod
    def train(
        data_loader,
        model,
        optimizer,
        device,
        scheduler=None,
        accumulation_steps=1
    ):
        
        losses = AverageMeter()
        predictions = []
        model.train()
        if accumulation_steps > 1:
            optimizer.zero_grad()
        #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
        tk0 = tqdm(data_loader, total=len(data_loader),disable=False)
        for b_idx, data in enumerate(tk0):
            for key, value in data.items():
                data[key] = value.to(device)
            if accumulation_steps == 1 and b_idx == 0:
                optimizer.zero_grad()
            _, loss = model(**data)
                   
            with torch.set_grad_enabled(True):
                loss.backward()
                if (b_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    if b_idx > 0:
                        optimizer.zero_grad()
            losses.update(loss.item(), data_loader.batch_size)
            tk0.set_postfix(loss=losses.avg)
        return losses.avg

    @staticmethod
    def evaluate(data_loader, model, device):
        losses = AverageMeter()
        final_predictions = []
        model.eval()
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader), disable=False)
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, loss = model(**data)
                predictions = predictions.cpu()
                losses.update(loss.item(), data_loader.batch_size)
                final_predictions.append(predictions)
                tk0.set_postfix(loss=losses.avg)
        return final_predictions, losses.avg

    @staticmethod
    def predict(data_loader, model, device, use_tpu=False):
        model.eval()
        final_predictions = []
        with torch.no_grad():
            #tk0 = tqdm(data_loader, total=len(data_loader), disable=use_tpu)
            tk0 = tqdm(data_loader, total=len(data_loader))
            for b_idx, data in enumerate(tk0):
                for key, value in data.items():
                    data[key] = value.to(device)
                predictions, _ = model(**data)
                predictions = predictions.cpu()
                final_predictions.append(predictions)
        return final_predictions