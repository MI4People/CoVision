import torch
from torchmetrics.classification import F1Score


def training_loop(
    model,
    num_classes,
    device,
    train_loader,
    optimizer,
    loss_function,
):

    device = device or torch.device("cpu")
    model.train()

    num_correct = 0
    num_total = 0
    running_loss = 0.0
    running_steps = 0
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="micro").to(device)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        num_total += target.size(0)
        num_correct += (predicted == target).sum().item()
        f1.update(predicted, target)

        running_loss += loss.item()
        running_steps += 1

    return (running_loss, num_correct, num_total, running_steps, f1.compute().item())


def val_loop(model, num_classes, device, val_loader, loss_function):
    device = device or torch.device("cpu")
    model.eval()

    num_correct = 0
    num_total = 0
    running_loss = 0.0
    running_steps = 0
    f1 = F1Score(task="multiclass", num_classes=num_classes, average="micro").to(device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = loss_function(outputs, target)

            _, predicted = torch.max(outputs.data, 1)
            num_total += target.size(0)
            num_correct += (predicted == target).sum().item()
            f1.update(predicted, target)

            running_loss += loss.item()
            running_steps += 1

    return (running_loss, num_correct, num_total, running_steps, f1.compute().item())
