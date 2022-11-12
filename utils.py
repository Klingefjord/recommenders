import torch
import matplotlib.pyplot as plt


def create_hook(layer, activations):
    def hook(_, __, output):
        activations[layer] = output.detach()

    return hook


def print_activations(model, x):
    """
    Perform a forward pass through the model and print the activations of each layer.
    """

    activations = {}
    model.eval()

    # register a forward hook
    layers = [
        m
        for m in model.modules()
        if not isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Dropout)
    ]
    for l in layers:
        l.register_forward_hook(create_hook(l, activations))

    # perform a forward pass
    model(x)

    # plot the activations
    plt.figure(figsize=(20, 4))
    legends = []

    for i, key in enumerate(activations):
        act = activations[key]
        print(
            "layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%"
            % (i, key, act.mean(), act.std(), (act.abs() > 0.97).float().mean() * 100)
        )
        hy, hx = torch.histogram(act, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f"layer {i} ({key}")

    plt.legend(legends)
    plt.title("activation distribution")
