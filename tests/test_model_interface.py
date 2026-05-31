import torch


def _batch(batch_size=2):
    images = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(0, 10, (batch_size,))
    return images, labels


def test_model_interface_runs_training_step_and_optimizer():
    from model import MInterface

    module = MInterface(
        model_name="example_net",
        loss="cross_entropy",
        metric="accuracy",
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
        lr_scheduler="none",
        num_classes=10,
        in_channels=1,
    )

    loss = module.training_step(_batch(), 0)
    optimizer = module.configure_optimizers()

    assert loss.ndim == 0
    assert loss.requires_grad
    assert isinstance(optimizer, torch.optim.Adam)
