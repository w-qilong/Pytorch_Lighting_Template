import torch


def test_data_interface_builds_fake_data_loaders(tmp_path):
    from data import DInterface

    dm = DInterface(
        train_dataset="example_data",
        val_datasets=["example_data"],
        test_datasets=["example_data"],
        data_dir=str(tmp_path),
        batch_size=2,
        num_workers=0,
        image_size=28,
        num_classes=10,
        num_samples=8,
    )

    dm.setup("fit")
    images, labels = next(iter(dm.train_dataloader()))

    assert images.shape == (2, 1, 28, 28)
    assert labels.dtype == torch.long
    assert len(dm.val_dataloader()) == 1

    dm.setup("test")
    assert len(dm.test_dataloader()) == 1
