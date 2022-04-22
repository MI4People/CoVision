import torch

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def training_loop(model, num_classes, device, train_loader, optimizer, loss_function, epoch, num_epochs):

    train_loss = []

    model.train()

    n_total_steps = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):

        if num_classes == 1:
            target = target.type(torch.DoubleTensor)
        else:
            target = target.type(torch.LongTensor)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            output = model(data)

            if num_classes == 1:
                loss = loss_function(torch.flatten(output), target)
            else:
                loss = loss_function(output, target)

            loss.backward()

            optimizer.step()

            if batch_idx % 2 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item():.4f}')

            train_loss.append(loss.item())
        
    return train_loss


def val_loop(model, num_classes, device, val_loader, loss_function):

    model.eval()

    with torch.no_grad():
        for i, (data,target) in enumerate(val_loader):
            
            if num_classes == 1:
                target = target.type(torch.DoubleTensor)
            else:
                target = target.type(torch.LongTensor)

            data, target = data.to(device), target.to(device)
            output = model(data)

            if num_classes == 1:
                curr_loss = loss_function(torch.flatten(output),target)
            else:
                curr_loss = loss_function(output, target)
            
            # Is this the correct solution for calculating the 
            output = torch.sigmoid(output)

            output = output.cpu().numpy() #this numpy array needs to look like this np([1, 0, 1, 0]) instead of np([0.2, -0.1, 0.8, 0.7])
            
            if num_classes == 1:

                for j in range(len(output)):
                    if output[j] > 0.5:
                        output[j] = 1
                    else:
                        output[j] = 0
            else:
                output = np.argmax(output, axis=1)

            if i==0:
                predictions = output
                targets = target.data.cpu().numpy()
                loss = np.array([curr_loss.data.cpu().numpy()])

            else:
                predictions = np.concatenate((predictions, output))
                targets = np.concatenate((targets, target.data.cpu().numpy()))
                loss = np.concatenate((loss, np.array([curr_loss.data.cpu().numpy()])))

    accuracy = accuracy_score(targets, predictions)
    conf_mat = confusion_matrix(targets, predictions)

    sensitivity = conf_mat.diagonal()/conf_mat.sum(axis=1)

    print("Val Accuracy: ", accuracy, "Val Sensitivity (Overall): ", np.mean(sensitivity), "Val loss: ", np.mean(loss))

    return targets, predictions, accuracy, np.mean(loss)