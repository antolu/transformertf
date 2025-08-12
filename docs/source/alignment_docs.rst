Sequence Alignment
==================

Understanding Encoder Alignment
-------------------------------

TransformerTF supports two sequence alignment strategies that control how variable-length sequences are padded and masked. This is essential for models using PyTorch's ``pack_padded_sequence`` functionality and attention mechanisms.

**Left Alignment**
    - Data positioned at start, padding at end
    - PyTorch ``pack_padded_sequence`` compatible
    - Used by LSTM-style models

**Right Alignment (Default)**
    - Data positioned at end, padding at start
    - Used by TFT-family models and encoder-decoder architectures
    - Default choice because time series data is often truncated from the left (earlier in time)
      to maintain continuity with decoder sequences that predict future values
    - Requires alignment conversion for PyTorch packing

Concrete Tensor Examples
------------------------

Given sequences with lengths ``[3, 2, 4]`` padded to max length ``5``:

**Left Alignment (encoder_alignment="left")**

.. code-block:: python

   import torch

   # Left-aligned sequences - data at start, padding at end
   sequences = torch.tensor([
       [[1., 2.], [2., 3.], [3., 4.], [0., 0.], [0., 0.]],  # length=3
       [[4., 5.], [5., 6.], [0., 0.], [0., 0.], [0., 0.]],  # length=2
       [[6., 7.], [7., 8.], [8., 9.], [9., 0.], [0., 0.]]   # length=4
   ])

   print(sequences[:, :, 0])  # First feature only
   # tensor([[1., 2., 3., 0., 0.],  ← data=[1,2,3], padding=[0,0]
   #         [4., 5., 0., 0., 0.],  ← data=[4,5], padding=[0,0,0]
   #         [6., 7., 8., 9., 0.]]) ← data=[6,7,8,9], padding=[0]

**Right Alignment (encoder_alignment="right", Default)**

.. code-block:: python

   from transformertf.utils.sequence import align_encoder_sequences

   # Default behavior: Convert to right alignment - data at end, padding at start
   right_aligned = align_encoder_sequences(sequences, lengths)  # alignment="right" is default

   print(right_aligned[:, :, 0])
   # tensor([[0., 0., 1., 2., 3.],  ← padding=[0,0], data=[1,2,3]
   #         [0., 0., 0., 4., 5.],  ← padding=[0,0,0], data=[4,5]
   #         [0., 6., 7., 8., 9.]]) ← padding=[0], data=[6,7,8,9]

Attention Masking
-----------------

The alignment strategy determines masking patterns:

**Left Alignment Masks**

.. code-block:: python

   from transformertf.models._base_transformer import create_mask

   lengths = torch.tensor([3, 2, 4])

   # Padding mask - True indicates padding positions
   left_padding_mask = create_mask(size=5, lengths=lengths, alignment="left")
   print(left_padding_mask)
   # tensor([[False, False, False,  True,  True],  ← padding at positions 3,4
   #         [False, False,  True,  True,  True],  ← padding at positions 2,3,4
   #         [False, False, False, False,  True]]) ← padding at position 4

   # Data mask - True indicates data positions
   left_data_mask = create_mask(size=5, lengths=lengths, alignment="left", inverse=True)
   print(left_data_mask)
   # tensor([[ True,  True,  True, False, False],  ← data at positions 0,1,2
   #         [ True,  True, False, False, False],  ← data at positions 0,1
   #         [ True,  True,  True,  True, False]]) ← data at positions 0,1,2,3

**Right Alignment Masks**

.. code-block:: python

   # Padding mask - True indicates padding positions
   right_padding_mask = create_mask(size=5, lengths=lengths, alignment="right")
   print(right_padding_mask)
   # tensor([[ True,  True, False, False, False],  ← padding at positions 0,1
   #         [ True,  True,  True, False, False],  ← padding at positions 0,1,2
   #         [ True, False, False, False, False]]) ← padding at position 0

   # Data mask - True indicates data positions
   right_data_mask = create_mask(size=5, lengths=lengths, alignment="right", inverse=True)
   print(right_data_mask)
   # tensor([[False, False,  True,  True,  True],  ← data at positions 2,3,4
   #         [False, False, False,  True,  True],  ← data at positions 3,4
   #         [False,  True,  True,  True,  True]]) ← data at positions 1,2,3,4

Configuration Examples
----------------------

**For PyTorch-Compatible Models**

Use left alignment for models that work directly with PyTorch's packed sequences:

.. code-block:: yaml

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       encoder_alignment: "left"  # Explicit left - data at start, padding at end
       train_df_paths: ["data.parquet"]
       target_covariate: "magnetic_field"
       known_covariates: ["voltage", "temperature"]

   model:
     class_path: transformertf.models.transformer_lstm.TransformerLSTM
     init_args:
       d_model: 64
       num_heads: 4

**For TFT-Family Models**

Use right alignment for Temporal Fusion Transformer and related models:

.. code-block:: yaml

   data:
     class_path: transformertf.data.EncoderDecoderDataModule
     init_args:
       encoder_alignment: "right"  # Default - data at end, padding at start
       train_df_paths: ["data.parquet"]
       target_covariate: "magnetic_field"
       known_covariates: ["voltage", "temperature"]

   model:
     class_path: transformertf.models.temporal_fusion_transformer.TemporalFusionTransformer
     init_args:
       d_model: 64
       num_heads: 4

PyTorch Compatibility
--------------------

**Left Alignment** works seamlessly with PyTorch's sequence processing:

.. code-block:: python

   from transformertf.utils.sequence import pack_encoder_sequences
   import torch.nn.utils.rnn as rnn_utils

   # Left-aligned sequences work directly
   sequences = torch.randn(3, 10, 64)  # (batch, seq_len, features)
   lengths = torch.tensor([8, 10, 6])

   # Pack sequences for efficient LSTM processing
   packed = pack_encoder_sequences(sequences, lengths, align_first=False)

   # Use with LSTM
   lstm = torch.nn.LSTM(64, 128, batch_first=True)
   packed_output, (h_n, c_n) = lstm(packed)

**Right Alignment** requires conversion for PyTorch compatibility:

.. code-block:: python

   # Right-aligned sequences need alignment conversion
   right_sequences = align_encoder_sequences(left_sequences, lengths)

   # Then pack (align_first=True handles the conversion)
   packed = pack_encoder_sequences(right_sequences, lengths, align_first=True)

Model Compatibility Guide
-------------------------

==========================================  ====================
Model                                       Recommended Alignment
==========================================  ====================
LSTM                                        left
EncoderDecoderLSTM                          left
TransformerLSTM                             left
Transformer                                 left
TemporalFusionTransformer                   right
PFTemporalFusionTransformer                 right
xTFT                                        right
TSMixer                                     left
==========================================  ====================

When to Use Each Alignment
--------------------------

**Use Left Alignment When:**
- Working with standard PyTorch sequence models
- Training LSTM or GRU architectures
- Need direct ``pack_padded_sequence`` compatibility
- Following standard sequence processing conventions

**Use Right Alignment When:**
- Working with TFT-family models
- Models expect data aligned to sequence end
- Implementing custom attention patterns that expect end-alignment
- Maintaining compatibility with pre-trained TFT checkpoints

The alignment choice affects model performance, so always match the alignment to your model's architectural expectations.
