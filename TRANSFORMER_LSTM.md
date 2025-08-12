# TransformerLSTM

**TransformerLSTM** is a hybrid deep learning architecture that combines Long Short-Term Memory (LSTM) networks with Transformer attention mechanisms for time series forecasting. It was developed as part of the TransformerTF framework for modeling temporal dynamics in physics applications, particularly magnetic field transfer functions at CERN particle accelerators.

## Architecture

TransformerLSTM follows a four-stage processing pipeline that builds upon traditional encoder-decoder architectures while incorporating modern attention mechanisms.

### Core Components

The model consists of the following key components:

1. **Gated Residual Networks (GRNs)** for input feature transformation
2. **LSTM Encoder-Decoder** for sequential processing  
3. **Custom Transformer Blocks** for attention-based refinement
4. **Linear Output Projection** for final predictions

### Data Flow

```
Input Sequences
     ↓
┌─────────────────┐    ┌─────────────────┐
│ past_sequence   │    │ future_sequence │
│ (B×Tp×Fp)      │    │ (B×Tf×Ff)      │
└─────────────────┘    └─────────────────┘
     ↓                          ↓
┌─────────────────┐    ┌─────────────────┐
│  Encoder GRN    │    │  Decoder GRN    │
└─────────────────┘    └─────────────────┘
     ↓                          ↓
┌─────────────────┐    ┌─────────────────┐
│ Encoder LSTM    │    │ Decoder LSTM    │
│ + Residual      │    │ + Residual      │
└─────────────────┘    └─────────────────┘
     ↓                          ↓
┌─────────────────────────────────────────┐
│           Transformer Blocks            │
│  ┌─────────────────────────────────┐    │
│  │    Encoder Self-Attention       │    │
│  │    + GateAddNorm               │    │
│  └─────────────────────────────────┘    │
│                   ↓                     │
│  ┌─────────────────────────────────┐    │
│  │      Cross-Attention            │    │
│  │  Query: decoder_output          │    │
│  │  K,V: encoder + decoder concat  │    │
│  │      + GateAddNorm             │    │
│  └─────────────────────────────────┘    │
│     (repeat for num_transformer_blocks) │
└─────────────────────────────────────────┘
                   ↓
          ┌─────────────────┐
          │ Linear Output   │
          │ (B×Tf×D_out)    │
          └─────────────────┘
```

Where:
- B = batch size
- Tp, Tf = past and future sequence lengths  
- Fp, Ff = past and future feature dimensions
- D_out = output dimension

## Technical Details

### Input/Output Specification

**Inputs:**
- `past_sequence`: Historical time series data of shape `(batch_size, past_seq_len, num_past_features)`
- `future_sequence`: Known future inputs of shape `(batch_size, future_seq_len, num_future_features)`  
- `encoder_lengths`, `decoder_lengths`: Optional sequence length tensors for variable-length sequence handling

**Output:**
- Predictions tensor of shape `(batch_size, future_seq_len, output_dim)`

### Transformer Block Architecture

The custom Transformer blocks differ significantly from standard Transformer layers. Each block processes encoder and decoder sequences separately:

1. **Encoder Self-Attention**: The encoder output attends to itself using interpretable multi-head attention with padding masks
2. **Cross-Attention**: The decoder queries the concatenation of both encoder and decoder outputs, providing full contextual access
3. **Gated Processing**: Both attention mechanisms are followed by `GateAddNorm` layers that combine gating, residual connections, and layer normalization

### Key Hyperparameters

- `d_model`: Hidden dimension for LSTMs and attention layers (typically 128-512)
- `num_layers`: Number of LSTM layers in encoder and decoder (typically 1-3)
- `num_transformer_blocks`: Depth of Transformer processing (typically 1-4)
- `num_heads`: Number of attention heads (typically 4-16)
- `causal_attention`: Boolean controlling future information masking
- `share_lstm_weights`: Whether encoder and decoder LSTMs share parameters

## Relationship to Existing Models

TransformerLSTM builds upon two established architectures in the TransformerTF framework:

### Compared to AttentionLSTM

The **AttentionLSTM** model represents a simpler attention-augmented approach, using a single attention layer applied after LSTM processing. TransformerLSTM extends this concept by:

- Applying multiple attention layers through stacked Transformer blocks
- Incorporating both self-attention and cross-attention mechanisms  
- Using more sophisticated gated processing with `GateAddNorm` layers
- Enabling iterative refinement of sequence representations

### Compared to Temporal Fusion Transformer

The **Temporal Fusion Transformer (TFT)** is a more complex architecture designed specifically for multi-variate time series with static covariates. Key differences include:

- TFT uses variable selection networks for automatic feature selection
- TFT explicitly processes static (time-invariant) features
- TFT applies attention after variable selection and static enrichment
- TransformerLSTM focuses on the temporal modeling aspects without specialized feature selection

## Design Rationale

### Hybrid Architecture Benefits

The combination of LSTM and Transformer components leverages complementary strengths:

- **LSTMs** provide strong inductive bias for sequential temporal data and natural handling of variable-length sequences
- **Transformers** enable modeling of complex long-range dependencies and parallel processing of sequences
- **Post-LSTM attention** operates on context-aware hidden states rather than raw embeddings

### Gating Mechanisms

Extensive use of gating throughout the architecture allows for:
- Controlled information flow through `GatedResidualNetwork` input processing
- Adaptive feature combination via `GateAddNorm` layers
- Non-linear transformations beyond simple linear projections

## Implementation Details

TransformerLSTM is implemented as a PyTorch Lightning module within the TransformerTF framework. Key implementation features include:

- Efficient variable-length sequence handling using `torch.nn.utils.rnn.pack_padded_sequence`
- Support for both causal and non-causal attention patterns
- Optional LSTM weight sharing between encoder and decoder
- Integration with the TransformerTF data pipeline and training infrastructure

## Applications

TransformerLSTM is particularly suited for time series forecasting tasks that require:
- Modeling of complex temporal dependencies
- Processing of moderate-complexity time series (more than simple patterns, less than highly multivariate data)
- Balance between model expressiveness and computational efficiency
- Sequential processing with attention-based refinement

The model was originally developed for modeling magnetic field transfer functions in particle accelerator systems, where temporal dynamics exhibit both sequential dependencies and complex long-range interactions.

## See Also

- [AttentionLSTM](transformertf/models/attention_lstm/) - Simpler attention-augmented LSTM
- [Temporal Fusion Transformer](transformertf/models/temporal_fusion_transformer/) - Complex multi-variate time series architecture
- [TransformerTF Framework](.) - Parent framework for physics-informed time series modeling