# EncoderDecoderLSTM

EncoderDecoderLSTM is a sequence-to-sequence deep learning architecture that uses two separate Long Short-Term Memory networks for time series forecasting. The model implements the classical encoder-decoder paradigm where an encoder LSTM compresses historical information into a context vector, which then initializes a decoder LSTM for generating future predictions. This architecture is part of the TransformerTF framework developed for temporal modeling in physics applications.

## Architecture

The EncoderDecoderLSTM follows the fundamental encoder-decoder design pattern, explicitly separating the processing of historical and future sequences through distinct LSTM networks. This separation allows the model to handle different feature sets for past and future time periods while learning specialized representations for each phase.

### Core Components

The model consists of four primary components that work together to process temporal sequences:

• Encoder LSTM processes historical time series data and creates a compressed representation
• Context transfer mechanism handles state projection when encoder and decoder dimensions differ  
• Decoder LSTM generates predictions using the encoder's compressed context
• Output head applies final transformations to produce predictions

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
│ num_enc_layers  │
│ d_encoder       │
└─────────────────┘
     ↓
┌─────────────────┐ (optional)
│ State Projection│
│ h_proj, c_proj  │
└─────────────────┘
     ↓                           ↓
┌─────────────────────────────────────┐
│         Decoder LSTM                │
│       num_dec_layers                │
│        d_decoder                    │
│                                     │
│ Initial State = Encoder Context     │
└─────────────────────────────────────┘
                   ↓
          ┌─────────────────┐
          │    MLP Head     │
          │  (configurable) │
          └─────────────────┘
                   ↓
          ┌─────────────────┐
          │ Final Output    │
          │ (B×Tf×D_out)    │
          └─────────────────┘
```

Where B represents batch size, Tp and Tf are past and future sequence lengths, Fp and Ff are past and future feature dimensions, and D_out is the output dimension.

## Technical Details

### Input and Output Specification

The model expects two input sequences with potentially different characteristics. The past_sequence contains historical time series data with shape (batch_size, past_seq_len, num_past_features), while the future_sequence provides known future features with shape (batch_size, future_seq_len, num_future_features). The model produces predictions as a tensor of shape (batch_size, future_seq_len, output_dim).

### Encoder-Decoder Mechanism

The core innovation lies in the separation of encoding and decoding phases. During the encoding phase, the encoder LSTM processes the entire past_sequence and produces final hidden state h_enc and cell state c_enc. These states form a context vector that encapsulates all historical information in a fixed-size representation.

The context transfer phase handles cases where encoder and decoder have different hidden dimensions. When d_encoder differs from d_decoder, linear projection layers transform the context states to match the decoder's requirements.

During the decoding phase, the decoder LSTM is initialized with this context vector and processes the future_sequence to generate predictions step by step. This design allows the model to leverage both the compressed historical context and known future features when making predictions.

### State Projection

When encoder and decoder networks have different hidden dimensions, the model employs linear projection layers to transform the encoder's final states:

```python
h_decoder_init = h_proj(h_enc)  # Transform to decoder dimensions
c_decoder_init = c_proj(c_enc)  # Transform to decoder dimensions
```

This flexibility allows for asymmetric architectures where the encoder and decoder can have different capacities based on the complexity of their respective tasks.

### Output Head Configuration

The output head can be configured in two ways depending on the complexity requirements. A simple linear projection provides direct mapping from decoder output to predictions, suitable for straightforward forecasting tasks. Alternatively, a multi-layer perceptron with configurable hidden layers, activation functions, and dropout can handle more complex output transformations.

### Key Hyperparameters

The model's behavior is controlled by several important hyperparameters:

• d_encoder and d_decoder control the hidden dimensions of the respective LSTM networks
• num_encoder_layers and num_decoder_layers determine the depth of each network
• output_dim specifies the final prediction dimension
• mlp_hidden_dims configures the hidden layer sizes for the MLP output head
• activation and dropout parameters control the non-linear transformations and regularization

## Relationship to Other Models

### Compared to Basic LSTM

The basic LSTM model in the TransformerTF framework represents a simpler, single-network approach to sequence modeling. While the basic LSTM processes a single continuous sequence and produces outputs at each timestep, the EncoderDecoderLSTM explicitly separates past and future sequence processing through two distinct networks.

The key difference lies in the information flow mechanism. The basic LSTM maintains sequential state evolution through one network, while the EncoderDecoderLSTM uses a context vector to transfer information from the encoder to decoder. This separation allows the EncoderDecoderLSTM to handle different feature sets for historical and future periods, making it more suitable for seq2seq forecasting tasks.

The EncoderDecoderLSTM also provides greater architectural flexibility, allowing different numbers of layers and hidden dimensions for the encoder and decoder based on the complexity of their respective tasks.

### Compared to Temporal Fusion Transformer

The Temporal Fusion Transformer represents a significantly more sophisticated approach to sequence-to-sequence modeling. While both architectures follow the encoder-decoder paradigm, TFT incorporates several advanced components that extend beyond the classical LSTM-based approach.

TFT includes variable selection networks that automatically weight feature importance, explicit processing of static time-invariant covariates, and multi-head attention mechanisms that replace the simple context transfer with rich attention-based information flow. The TFT also employs gated residual networks for complex non-linear transformations throughout the architecture.

In contrast, EncoderDecoderLSTM implements the classical RNN-based seq2seq paradigm with a focus on simplicity and interpretability. The model's context vector mechanism creates a clear information bottleneck that forces the encoder to learn compressed representations of the historical sequence.

## Mathematical Formulation

### Encoding Phase
For a past sequence (x₁, x₂, ..., x_T_past), the encoder computes hidden and cell states at each timestep:

h_t^enc, c_t^enc = LSTM_enc(x_t, h_{t-1}^enc, c_{t-1}^enc)

The context vector is formed by the final encoder states: context = (h_T_past^enc, c_T_past^enc)

### State Projection
When encoder and decoder dimensions differ, linear transformations project the context:

h₀^dec = W_h · h_T_past^enc + b_h
c₀^dec = W_c · c_T_past^enc + b_c

### Decoding Phase  
Using the context as initial state and processing the future sequence (x'₁, x'₂, ..., x'_T_future):

h_t^dec, c_t^dec = LSTM_dec(x'_t, h_{t-1}^dec, c_{t-1}^dec)
y_t = MLP(h_t^dec)

## Design Rationale

### Teacher Forcing Strategy
During training, the model implements teacher forcing by using ground-truth future_sequence as decoder input. This technique stabilizes the learning process by providing the decoder with accurate inputs at each step, though it requires known future features even during inference.

### Context Compression
The encoder's final state must compress all historical information into a fixed-size vector. This design choice creates a powerful inductive bias for learning meaningful temporal representations, though it may create an information bottleneck for very long sequences where important early information might be lost.

### Modular Architecture
The separation of encoder and decoder networks allows for independent architectural choices for each component. This modularity enables different scaling strategies for encoder and decoder complexity and provides clear separation between the historical analysis and future generation phases.

## Implementation Details

EncoderDecoderLSTM is implemented as a PyTorch Lightning module with several technical features that enhance its practical utility. The model supports variable-length sequences through packed sequence processing, enabling efficient batch handling of sequences with different lengths. Memory efficiency is maintained through proper LSTM state management and gradient flow optimization.

The implementation provides full integration with the TransformerTF data pipeline and training infrastructure, along with flexible MLP output heads that can be configured for different prediction requirements. The model also includes proper handling of sequence masking and padding for robust batch processing.

## Applications

EncoderDecoderLSTM is particularly well-suited for time series forecasting scenarios where historical and future features have different characteristics. The model excels in situations requiring clear separation between analysis and prediction phases, especially when future covariates are available such as weather forecasts, calendar features, or planned interventions.

The architecture is ideal for problems where the traditional seq2seq modeling approach is appropriate, providing interpretable modeling phases through its explicit encoder-decoder separation. Within the context of CERN's particle accelerator systems, the model effectively handles scenarios where historical measurements and future operational parameters have distinct characteristics and require specialized processing approaches.

## See Also

• LSTM - Basic LSTM architecture for comparison
• Temporal Fusion Transformer - Advanced attention-based seq2seq architecture  
• TransformerTF Framework - Parent framework for physics-informed time series modeling