.. _usage:

Usage
=====


Transfer learning
------------------

For transfer learning using the LightningCLI method, add key :code:`transfer_ckpt` to the config YAML file at the top level, pointing to the checkpoint file to be used for transfer learning. The checkpoint file should be a PyTorch Lightning checkpoint file, with the model state dict saved as the key :code:`state_dict`.
When doing transfer learning, the datamodule is re-created from the new dataset, while the model is loaded from the checkpoint file. The model is then trained on the new dataset, with the datamodule being passed to the model for training.

Transfer learning is currently only supported for the TemporalFusionTransformer model.
