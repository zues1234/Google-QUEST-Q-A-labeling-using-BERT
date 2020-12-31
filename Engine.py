def train_fn(model, dataloader, optimizer, device, scheduler=None):
    model.train()
    for batch_id, data in enumerate(dataloader):
        ids = (data["ids"]).to(device, dtype=torch.long)
        mask = (data["mask"]).to(device, dtype=torch.long)
        token = (data["token"]).to(device, dtype=torch.long)
        target = (data["targets"]).to(device, dtype=torch.float)

        optimizer.zero_grad()
        output = model(ids=ids, attention_mask=mask, token_type_ids=token)
        loss = loss_fn(output, targets=target)
        loss.backward()
        xm.optimizer_step(optimizer)

        if scheduler is not None:
            scheduler.step()
        
        if batch_id % 10 == 0:
            xm.master_print(f"batch = {batch_id}, loss = {loss}")


def eval_fn(model, dataloader, device):
    model.eval()
    targets=[]
    outputs=[]
    for batch_id, data in enumerate(dataloader):
        ids = (data["ids"]).to(device, dtype=torch.long)
        mask = (data["mask"]).to(device, dtype=torch.long)
        token = (data["token"]).to(device, dtype=torch.long)
        target = (data["targets"]).to(device, dtype=torch.float)

        output = model(ids=ids, attention_mask=mask, token_type_ids=token)
        loss = loss_fn(output, targets=target)

        targets.append(target.cpu().detach().numpy())
        outputs.append(output.cpu().detach().numpy())

        target = np.vstack(targets)
        output =  np.vstack(outputs)

        return output, target