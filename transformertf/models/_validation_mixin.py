from __future__ import annotations

from ..utils.sequence import validate_encoder_alignment


class EncoderAlignmentValidationMixin:
    """Mixin for validating encoder alignment by examining tensor structure."""

    def validate_encoder_alignment_with_batch(self, batch) -> None:
        """
        Validate encoder alignment using actual batch data.

        Parameters
        ----------
        batch : dict
            Batch containing encoder_input and encoder_lengths.
        """
        if not hasattr(self, "trainer") or self.trainer is None:
            return

        if not hasattr(self.trainer, "datamodule") or self.trainer.datamodule is None:
            return

        datamodule = self.trainer.datamodule
        if (
            not hasattr(datamodule, "hparams")
            or "encoder_alignment" not in datamodule.hparams
        ):
            return

        encoder_alignment = datamodule.hparams["encoder_alignment"]
        encoder_input = batch.get("encoder_input")
        encoder_lengths = batch.get("encoder_lengths")

        if encoder_input is not None and encoder_lengths is not None:
            # Handle shape (batch_size, 1) -> (batch_size,)
            if encoder_lengths.dim() > 1:
                encoder_lengths = encoder_lengths.squeeze(-1)

            validate_encoder_alignment(
                encoder_input, encoder_lengths, encoder_alignment
            )
