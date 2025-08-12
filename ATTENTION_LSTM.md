# AttentionLSTM

AttentionLSTM is a sequence-to-sequence deep learning architecture that adds multi-head self-attention to the standard LSTM encoder-decoder framework for time series forecasting. The model sits between simpler encoder-decoder approaches and more complex transformer-based architectures, combining the sequential processing of LSTMs with attention mechanisms that can model long-range dependencies. This architecture is part of the TransformerTF framework developed for temporal modeling in physics applications.

## Architecture

The AttentionLSTM implements a design that combines recurrent and attention-based processing. The architecture maintains the encoder-decoder structure while introducing attention mechanisms that allow the decoder to focus on different parts of the input sequence.

### Core Components

The model consists of five primary components:

• Encoder LSTM processes historical time series data
• Decoder LSTM generates future predictions initialized with encoder context
• Multi-head self-attention mechanism captures dependencies across time steps
• Skip connection combines attention output with decoder states
• Output projection layer produces final predictions

### Data Flow

```
Historical Data          Known Future Features
     ↓                           ↓
┌─────────────────┐     ┌─────────────────┐
│ past_sequence   │     │ future_sequence │
│ (B×Tp×Fp)      │     │ (B×Tf×Ff)      │
└─────────────────┘     └─────────────────┘
     ↓
┌─────────────────┐
│ Encoder LSTM    │
│ num_layers      │
│ d_model         │
└─────────────────┘
     ↓ (final states)           ↓
┌─────────────────────────────────────┐
│         Decoder LSTM                │
│       (initialized with             │
│        encoder states)              │
└─────────────────────────────────────┘
     ↓                           ↓
┌─────────────────────────────────────┐
│     Multi-Head Attention            │
│                                     │
│  Query: decoder_output              │
│  Key/Value: concat(encoder_output,  │
│                   decoder_output)   │
└─────────────────────────────────────┘
                   ↓
          ┌─────────────────┐
          │  Skip Connection │
          │ (GateAddNorm or │
          │  LayerNorm)     │
          └─────────────────┘
                   ↓
          ┌─────────────────┐
          │ Linear Output   │
          │ (B×Tf×D_out)    │
          └─────────────────┘
```

Where B represents batch size, Tp and Tf are past and future sequence lengths, Fp and Ff are past and future feature dimensions, and D_out is the output dimension.

## Technical Details

### Input and Output Specification

The model processes two input sequences. The past_sequence contains historical time series data with shape (batch_size, past_seq_len, num_past_features), while the future_sequence provides known future features with shape (batch_size, future_seq_len, num_future_features). Optional length tensors handle variable-length sequences within batches. The model produces predictions as a tensor of shape (batch_size, future_seq_len, output_dim).

### Attention Mechanism Design

The multi-head self-attention operates on the decoder output. The attention mechanism uses the decoder output as queries while the keys and values are formed by concatenating both encoder and decoder output sequences. This allows the decoder to attend to both past observations and its own generated outputs at each time step.

The attention computation follows standard multi-head attention but uses this concatenated key-value construction. The mechanism can apply causal masking to prevent the decoder from attending to future positions in its own output sequence.

### Skip Connection and Gating

The model uses a skip connection mechanism that combines the attention output with the original decoder output. When use_gating is enabled, the model uses a GateAddNorm layer that applies a gated linear unit to control information flow, followed by residual connection and layer normalization. When gating is disabled, a simpler additive residual connection with layer normalization is used.

The gating mechanism provides control over the influence of the attention mechanism, allowing the model to balance between recurrent processing and attention-based refinement.

### Key Hyperparameters

The model's behavior is controlled by several hyperparameters:

• d_model controls the hidden dimension of LSTM networks and attention mechanisms
• num_layers determines the depth of both encoder and decoder LSTM networks
• num_heads specifies the number of attention heads
• use_gating enables or disables the gated skip connection mechanism
• causal_attention controls whether the decoder can attend to future positions
• output_dim specifies the final prediction dimension

## Position as Middle Ground Architecture

### Compared to EncoderDecoderLSTM

The EncoderDecoderLSTM relies solely on the encoder's final hidden state to transfer information to the decoder. This creates a potential information bottleneck where all historical information must be compressed into fixed-size vectors.

AttentionLSTM addresses this limitation by allowing the decoder to access the complete encoder output sequence through attention mechanisms. This helps with long sequences where early information might otherwise be lost in the compression process. The attention mechanism also provides interpretability by revealing which parts of the input sequence are most relevant for each prediction.

### Compared to TransformerLSTM

TransformerLSTM applies multiple transformer blocks with both self-attention and cross-attention mechanisms after LSTM processing. While this provides greater modeling capacity, it also increases computational requirements and architectural complexity.

AttentionLSTM achieves many of the benefits of attention-based processing through a single attention layer. This approach maintains efficiency while capturing important attention-based interactions. The model retains inductive bias for sequential data through its LSTM foundation while adding attention where it provides benefit.

The AttentionLSTM also provides clearer architectural interpretation, where the roles of sequential processing (LSTM) and relationship modeling (attention) are clearly separated and combined through the skip connection mechanism.

## Mathematical Formulation

### Encoding Phase
For a past sequence (x₁, x₂, ..., x_T_past), the encoder processes each timestep:

H_enc, (h_n, c_n) = LSTM_encoder(X_past)

where H_enc contains the output at each timestep and (h_n, c_n) represents the final hidden and cell states.

### Decoding Phase
The decoder is initialized with the encoder's final states and processes the future sequence:

H_dec, _ = LSTM_decoder(X_future, initial_state=(h_n, c_n))

### Attention Mechanism
The multi-head attention operates on the decoder output with concatenated encoder-decoder context:

Context = concat(H_enc, H_dec)
Attn_out = MultiHeadAttention(Query=H_dec, Key=Context, Value=Context)

### Skip Connection and Output
The final output combines attention and decoder representations:

Combined = SkipConnection(Attn_out, H_dec)
Output = Linear(Combined)

where SkipConnection can be either GateAddNorm or simple residual connection with layer normalization.

## Design Rationale

### Hybrid Processing Strategy
The combination of LSTM and attention mechanisms uses different strengths. LSTMs provide inductive bias for sequential temporal data and handle variable-length sequences naturally, while attention mechanisms enable modeling of long-range dependencies and non-sequential relationships.

### Selective Attention Application
Rather than replacing the LSTM structure entirely, AttentionLSTM applies attention where it provides benefit. This approach maintains computational efficiency while capturing important attention-based interactions that improve prediction quality.

### Interpretability
The attention mechanism provides interpretability by revealing which parts of the input sequence are most influential for each prediction. The attention weights can be visualized to understand the model's decision-making process, which is valuable in physics applications where understanding model behavior is important.

## Implementation Details

AttentionLSTM is implemented as a PyTorch Lightning module with several technical features. The model supports variable-length sequences through packed sequence processing and proper attention masking, enabling efficient batch handling of sequences with different lengths.

The implementation includes dropout at multiple levels, including within LSTM layers, attention mechanisms, and skip connections. The model provides full integration with the TransformerTF data pipeline and training infrastructure, along with configurable attention and gating mechanisms.

The architecture maintains memory efficiency through careful state management and supports both causal and non-causal attention patterns depending on the forecasting requirements.

## Applications

AttentionLSTM works well for time series forecasting scenarios that require both sequential processing and the ability to model temporal relationships. The model works when simple encoder-decoder approaches are insufficient but full transformer architectures may be unnecessarily complex.

The architecture is suitable for scenarios with moderate sequence lengths where the attention mechanism can capture important temporal relationships without the computational overhead of more complex attention-based architectures. The interpretability provided by attention weights makes it useful for applications where understanding model decisions is important.

Within the context of CERN's particle accelerator systems, AttentionLSTM handles magnetic field transfer function modeling where both sequential temporal dynamics and selective attention to specific historical periods are important for accurate predictions.

## See Also

• EncoderDecoderLSTM - Simpler encoder-decoder architecture without attention mechanisms
• TransformerLSTM - More complex hybrid architecture with multiple transformer blocks
• Temporal Fusion Transformer - Advanced attention-based architecture with variable selection
• TransformerTF Framework - Parent framework for physics-informed time series modeling