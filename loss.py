def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)