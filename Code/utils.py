import torch

def train(model, train_loader, optimizer, loss_eval, epoch):
    total_loss = 0
    total_train = 0
    correct_train = 0
    losses = []  # List to store losses
    model.train()

    for images, labels in train_loader:
        optimizer.zero_grad()  # every time train we want to gradients to be set to zero
        output = model(images)  # making the forward pass through the model
        loss = loss_eval(output, labels)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

        # accuracy
        _, predicted = torch.max(output.data, 1)  # we check the label which has maximum probability
        total_train += labels.size(0)
        correct_train += predicted.eq(labels.data).sum().item()

        print(f'Train Epoch: {epoch}: Accuracy:{correct_train/total_train}\tLoss: {loss.item()}')

        losses.append(loss.item())

    train_accuracy = 100 * correct_train / total_train

    return (train_accuracy, total_loss / len(train_loader))

def val_eval(model, val_loader, loss_func):
    model.eval()
    total_loss = 0
    total_val = 0
    correct_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = loss_func(output, labels)

            total_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total_val += labels.size(0)
            correct_val += predicted.eq(labels.data).sum().item()

    val_accuracy = 100 * correct_val / total_val

    return val_accuracy, total_loss / len(val_loader)


def test_eval(model, test_loader, optimizer, loss_eval, epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = model(images)  # forward pass
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)  # to get the total number of labels
                correct += (
                            predicted == labels).sum().item()

                print(f'Test Epoch: {epoch}: Accuracy:{correct / total}')

    acc = 100 * correct / total
    return acc