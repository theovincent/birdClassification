def validation(model, loss, loader, use_cuda):
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        validation_loss += loss(output, target).data.item()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(loader.dataset)

    print("\n\nValidation set:")
    print(
        f"Average loss: {validation_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({int(100.0 * correct / len(loader.dataset))}%)"
    )
