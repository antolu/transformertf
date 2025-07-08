from __future__ import annotations

import builtins
import typing

import numpy as np
import torch

arr_t = typing.TypeVar("arr_t", torch.Tensor, np.ndarray)
T_co = typing.TypeVar(
    "T_co",
    dict,
    tuple,
    typing.Sequence,
    torch.Tensor,
    covariant=True,
)
M_co = typing.TypeVar(
    "M_co",
    torch.Tensor,
    tuple[np.ndarray, ...],
    tuple[torch.Tensor, ...],
    list[np.ndarray],
    list[torch.Tensor],
    typing.Sequence[np.ndarray],
    typing.Sequence[torch.Tensor],
    covariant=True,
)
D_co = typing.TypeVar(
    "D_co",
    dict[typing.Hashable, np.ndarray],
    dict[typing.Hashable, torch.Tensor],
    covariant=True,
)


# pyright: reportGeneralTypeIssues=false
def _op_T(  # noqa: N802
    data: T_co, op: typing.Callable[[torch.Tensor], torch.Tensor]
) -> T_co:
    if isinstance(data, dict):
        return type(data)({k: _op_T(v, op) for k, v in data.items()})
    if isnamedtupleinstance(data):
        return type(data)(*[_op_T(v, op) for v in data])
    if isinstance(data, list | tuple):
        return type(data)([_op_T(v, op) for v in data])
    if isinstance(data, typing.Sequence):
        return type(data)([_op_T(v, op) for v in data])  # type: ignore[call-arg]
    if isinstance(data, torch.Tensor):
        return op(data)
    msg = f"Unknown type {type(data)}."
    raise TypeError(msg)


def detach(data: T_co) -> T_co:
    """
    Detach tensor data from the computational graph recursively.

    This function recursively traverses nested data structures (dicts, tuples,
    lists, sequences) and detaches all PyTorch tensors from their computational
    graphs. This is useful for preventing gradient computation and memory leaks
    when storing intermediate results or moving data between different contexts.

    Parameters
    ----------
    data : T_co
        The data structure to detach. Can be a tensor, dict, tuple, list,
        sequence, or any nested combination of these types containing tensors.

    Returns
    -------
    T_co
        The same data structure with all tensors detached from the
        computational graph. The structure and tensor values are preserved,
        but gradients will not flow through the detached tensors.

    Notes
    -----
    This function is commonly used in training loops when you need to:

    - Store intermediate results without keeping gradient information
    - Move data between different model contexts
    - Prevent memory leaks from accumulated gradients
    - Prepare data for logging or visualization

    The function preserves the exact structure of the input data and works
    with arbitrarily nested data structures.

    Examples
    --------
    Detach a single tensor:

    >>> import torch
    >>> from transformertf.utils.ops import detach
    >>>
    >>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = x ** 2
    >>> y_detached = detach(y)
    >>> print(y_detached.requires_grad)
    False

    Detach a dictionary of tensors:

    >>> data = {
    ...     "loss": torch.tensor(0.5, requires_grad=True),
    ...     "predictions": torch.randn(10, 3, requires_grad=True)
    ... }
    >>> detached_data = detach(data)
    >>> print(detached_data["loss"].requires_grad)
    False

    Detach nested structures:

    >>> nested_data = {
    ...     "batch": {
    ...         "input": torch.randn(32, 128, requires_grad=True),
    ...         "target": torch.randn(32, 10, requires_grad=True)
    ...     },
    ...     "metadata": [torch.tensor([1, 2, 3])]
    ... }
    >>> detached_nested = detach(nested_data)
    >>> print(detached_nested["batch"]["input"].requires_grad)
    False

    Use in training loop:

    >>> for batch in dataloader:
    ...     output = model(batch)
    ...     loss = criterion(output, batch["target"])
    ...
    ...     # Store results without gradients
    ...     results = detach({
    ...         "loss": loss,
    ...         "predictions": output,
    ...         "targets": batch["target"]
    ...     })
    ...
    ...     loss.backward()
    ...     optimizer.step()

    See Also
    --------
    to_cpu : Move data to CPU
    to : Move data to specific device
    torch.Tensor.detach : PyTorch's tensor detach method
    """
    return _op_T(data, lambda x: x.detach())


def to(data: T_co, device: torch.device) -> T_co:
    """
    Move tensor data to the specified device recursively.

    This function recursively traverses nested data structures and moves all
    PyTorch tensors to the specified device (CPU, GPU, etc.). It preserves
    the exact structure of the input data while ensuring all tensors are
    on the target device.

    Parameters
    ----------
    data : T_co
        The data structure to move. Can be a tensor, dict, tuple, list,
        sequence, or any nested combination of these types containing tensors.
    device : torch.device
        The target device to move tensors to. Can be created using
        `torch.device("cpu")`, `torch.device("cuda")`, or similar.

    Returns
    -------
    T_co
        The same data structure with all tensors moved to the specified device.
        The structure and tensor values are preserved, but tensors will be
        on the target device.

    Notes
    -----
    This function is essential for:

    - Moving data between CPU and GPU for training/inference
    - Ensuring data and model are on the same device
    - Preparing data for device-specific operations
    - Optimizing memory usage by moving data to appropriate devices

    The function handles device transfers efficiently and preserves
    gradient information if tensors require gradients.

    Examples
    --------
    Move data to GPU:

    >>> import torch
    >>> from transformertf.utils.ops import to
    >>>
    >>> # Single tensor
    >>> x = torch.randn(10, 3)
    >>> x_gpu = to(x, torch.device("cuda"))
    >>> print(x_gpu.device)
    cuda:0

    Move dictionary of tensors:

    >>> data = {
    ...     "input": torch.randn(32, 128),
    ...     "target": torch.randn(32, 10)
    ... }
    >>> data_gpu = to(data, torch.device("cuda"))
    >>> print(data_gpu["input"].device)
    cuda:0

    Move nested structures:

    >>> batch = {
    ...     "encoder_input": torch.randn(32, 100),
    ...     "decoder_input": torch.randn(32, 50),
    ...     "metadata": [torch.tensor([1, 2, 3])]
    ... }
    >>> batch_gpu = to(batch, torch.device("cuda"))
    >>> print(batch_gpu["encoder_input"].device)
    cuda:0

    Use in training loop:

    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> for batch in dataloader:
    ...     batch = to(batch, device)
    ...     output = model(batch["input"])
    ...     loss = criterion(output, batch["target"])

    Move data back to CPU for logging:

    >>> predictions_cpu = to(predictions, torch.device("cpu"))
    >>> # Now safe to convert to numpy or save

    See Also
    --------
    to_cpu : Convenience function to move data to CPU
    detach : Detach data from computational graph
    torch.Tensor.to : PyTorch's tensor device transfer method
    """
    return _op_T(data, lambda x: x.to(device))


def to_cpu(data: T_co) -> T_co:
    """
    Move tensor data to CPU recursively.

    This function is a convenience wrapper around `to` that specifically moves
    all tensors in nested data structures to CPU. It's commonly used when
    transferring data from GPU to CPU for logging, saving, or further processing.

    Parameters
    ----------
    data : T_co
        The data structure to move to CPU. Can be a tensor, dict, tuple, list,
        sequence, or any nested combination of these types containing tensors.

    Returns
    -------
    T_co
        The same data structure with all tensors moved to CPU. The structure
        and tensor values are preserved, but tensors will be on CPU.

    Notes
    -----
    This function is particularly useful for:

    - Preparing data for conversion to NumPy arrays
    - Moving results from GPU to CPU for logging or visualization
    - Reducing GPU memory usage by moving unused data to CPU
    - Preparing data for serialization or saving

    The function preserves gradient information if tensors require gradients,
    but typically you'll want to use `detach` as well if you don't need gradients.

    Examples
    --------
    Move single tensor to CPU:

    >>> import torch
    >>> from transformertf.utils.ops import to_cpu
    >>>
    >>> x = torch.randn(10, 3, device="cuda")
    >>> x_cpu = to_cpu(x)
    >>> print(x_cpu.device)
    cpu

    Move model outputs to CPU for logging:

    >>> model_output = {
    ...     "loss": torch.tensor(0.5, device="cuda"),
    ...     "predictions": torch.randn(32, 10, device="cuda"),
    ...     "hidden_states": torch.randn(32, 128, device="cuda")
    ... }
    >>> output_cpu = to_cpu(model_output)
    >>> print(output_cpu["loss"].device)
    cpu

    Convert to NumPy after moving to CPU:

    >>> predictions_gpu = torch.randn(100, 10, device="cuda")
    >>> predictions_cpu = to_cpu(predictions_gpu)
    >>> predictions_numpy = predictions_cpu.numpy()

    Use in prediction loop:

    >>> results = []
    >>> for batch in dataloader:
    ...     with torch.no_grad():
    ...         output = model(batch)
    ...         # Move to CPU and detach for storage
    ...         output_cpu = to_cpu(detach(output))
    ...         results.append(output_cpu)

    Common pattern for logging:

    >>> # During training
    >>> loss = criterion(output, target)
    >>>
    >>> # Log scalar value
    >>> logger.log("loss", to_cpu(detach(loss)).item())
    >>>
    >>> # Log predictions for analysis
    >>> pred_cpu = to_cpu(detach(output))
    >>> logger.log_predictions(pred_cpu)

    See Also
    --------
    to : Move data to specific device
    detach : Detach data from computational graph
    torch.Tensor.cpu : PyTorch's tensor CPU method
    """
    return _op_T(data, lambda x: x.to(torch.device("cpu")))


def truncate(data: M_co, length: int) -> M_co:
    """
    Truncate tensor data to specified length recursively.

    This function recursively traverses nested data structures and truncates
    all tensors to the specified length along the first dimension. It's useful
    for limiting sequence lengths, batch sizes, or dataset sizes.

    Parameters
    ----------
    data : M_co
        The data structure to truncate. Can be a tensor, tuple, list, sequence,
        or any nested combination of these types containing tensors.
    length : int
        The maximum length to truncate to. Tensors will be sliced as `x[:length]`.

    Returns
    -------
    M_co
        The same data structure with all tensors truncated to the specified length.
        The structure is preserved, but tensors will have at most `length` elements
        in their first dimension.

    Notes
    -----
    This function is commonly used for:

    - Limiting sequence lengths in NLP tasks
    - Reducing batch sizes for memory management
    - Creating smaller datasets for debugging
    - Ensuring consistent data sizes across batches

    The function only truncates along the first dimension (dimension 0).
    If tensors are shorter than the specified length, they remain unchanged.

    Examples
    --------
    Truncate single tensor:

    >>> import torch
    >>> from transformertf.utils.ops import truncate
    >>>
    >>> x = torch.randn(100, 64)
    >>> x_truncated = truncate(x, 50)
    >>> print(x_truncated.shape)
    torch.Size([50, 64])

    Truncate batch data:

    >>> batch = {
    ...     "input": torch.randn(32, 128),
    ...     "target": torch.randn(32, 10)
    ... }
    >>> small_batch = truncate(batch, 16)
    >>> print(small_batch["input"].shape)
    torch.Size([16, 128])

    Truncate sequence data:

    >>> sequences = [
    ...     torch.randn(200, 64),
    ...     torch.randn(150, 64),
    ...     torch.randn(300, 64)
    ... ]
    >>> truncated_sequences = truncate(sequences, 100)
    >>> print([seq.shape[0] for seq in truncated_sequences])
    [100, 100, 100]

    Use for debugging with smaller batches:

    >>> # During development, use smaller batches
    >>> for batch in dataloader:
    ...     debug_batch = truncate(batch, 4)  # Only 4 samples
    ...     output = model(debug_batch["input"])
    ...     loss = criterion(output, debug_batch["target"])

    Limit sequence length in NLP:

    >>> text_batch = {
    ...     "input_ids": torch.randint(0, 1000, (32, 512)),
    ...     "attention_mask": torch.ones(32, 512)
    ... }
    >>> # Limit to 256 tokens
    >>> limited_batch = truncate(text_batch, 256)
    >>> print(limited_batch["input_ids"].shape)
    torch.Size([32, 256])

    See Also
    --------
    slice : Apply general slicing to data
    concatenate : Concatenate data structures
    """
    return _op_T(data, lambda x: x[:length])


def squeeze(data: T_co) -> T_co:
    """
    Remove singleton dimensions from tensor data recursively.

    This function recursively traverses nested data structures and applies
    the squeeze operation to all tensors, removing dimensions of size 1.
    It's useful for cleaning up tensor shapes after operations that introduce
    singleton dimensions.

    Parameters
    ----------
    data : T_co
        The data structure to squeeze. Can be a tensor, dict, tuple, list,
        sequence, or any nested combination of these types containing tensors.

    Returns
    -------
    T_co
        The same data structure with all tensors squeezed to remove singleton
        dimensions. The structure is preserved, but tensors will have dimensions
        of size 1 removed.

    Notes
    -----
    This function is commonly used for:

    - Cleaning up tensor shapes after unsqueezing operations
    - Removing batch dimensions of size 1
    - Preparing tensors for operations that don't expect singleton dimensions
    - Normalizing tensor shapes across different model outputs

    The squeeze operation removes all dimensions of size 1. If you need to
    remove specific dimensions, use tensor slicing or indexing instead.

    Examples
    --------
    Squeeze single tensor:

    >>> import torch
    >>> from transformertf.utils.ops import squeeze
    >>>
    >>> x = torch.randn(1, 64, 1, 128)
    >>> x_squeezed = squeeze(x)
    >>> print(x_squeezed.shape)
    torch.Size([64, 128])

    Squeeze batch data:

    >>> batch = {
    ...     "predictions": torch.randn(1, 10, 1),
    ...     "targets": torch.randn(1, 10)
    ... }
    >>> squeezed_batch = squeeze(batch)
    >>> print(squeezed_batch["predictions"].shape)
    torch.Size([10])

    Squeeze model outputs:

    >>> model_output = {
    ...     "logits": torch.randn(1, 1, 1000),
    ...     "hidden": torch.randn(1, 512)
    ... }
    >>> clean_output = squeeze(model_output)
    >>> print(clean_output["logits"].shape)
    torch.Size([1000])

    Use after unsqueezing operations:

    >>> x = torch.randn(64, 128)
    >>> x_unsqueezed = x.unsqueeze(0).unsqueeze(2)  # Add dims
    >>> print(x_unsqueezed.shape)
    torch.Size([1, 64, 1, 128])
    >>> x_clean = squeeze(x_unsqueezed)
    >>> print(x_clean.shape)
    torch.Size([64, 128])

    Clean up prediction outputs:

    >>> # Model returns predictions with extra dimensions
    >>> raw_predictions = model(input_data)  # Shape: (1, seq_len, 1, vocab_size)
    >>> clean_predictions = squeeze(raw_predictions)  # Shape: (seq_len, vocab_size)
    >>> probabilities = torch.softmax(clean_predictions, dim=-1)

    See Also
    --------
    torch.Tensor.squeeze : PyTorch's tensor squeeze method
    torch.Tensor.unsqueeze : Add singleton dimensions
    """
    return _op_T(data, lambda x: x.squeeze())


@typing.overload
def concatenate(
    value: tuple[arr_t, ...] | list[arr_t] | typing.Sequence[arr_t],
) -> arr_t: ...


@typing.overload
def concatenate(value: typing.Sequence[tuple[arr_t, ...]]) -> tuple[arr_t, ...]: ...


@typing.overload  # type: ignore[misc]
def concatenate(value: typing.Sequence[dict[str, arr_t]]) -> dict[str, arr_t]: ...


@typing.overload
def concatenate(value: typing.Sequence[list[arr_t]]) -> list[arr_t]: ...


@typing.overload
def concatenate(
    value: typing.Sequence[typing.Sequence[arr_t]],
) -> typing.Sequence[arr_t]: ...


def concatenate(  # type: ignore
    value: (
        list[tuple[arr_t, ...]]
        | list[arr_t]
        | typing.Sequence[dict[str, arr_t]]
        | typing.Sequence[arr_t]
        | typing.Sequence[tuple[arr_t, ...]]
        | typing.Sequence[list[arr_t]]
        | typing.Sequence[typing.Sequence[arr_t]]
    ),
) -> (
    arr_t | dict[str, arr_t] | list[arr_t] | tuple[arr_t, ...] | typing.Sequence[arr_t]
):
    """
    Concatenate arrays or nested data structures along the first dimension.

    This function concatenates sequences of arrays (PyTorch tensors or NumPy arrays)
    or nested data structures containing arrays. It automatically handles different
    data structure types and preserves the structure while concatenating the arrays.

    Parameters
    ----------
    value : Sequence of arrays or data structures
        The sequence of data to concatenate. Can be:

        - Sequence of arrays (tensors/ndarrays)
        - Sequence of dictionaries with array values
        - Sequence of tuples containing arrays
        - Sequence of lists containing arrays
        - Sequence of other sequences containing arrays

        All elements must be of the same type and structure.

    Returns
    -------
    arr_t | dict[str, arr_t] | list[arr_t] | tuple[arr_t, ...] | typing.Sequence[arr_t]
        The concatenated result. The return type matches the input structure:

        - Arrays: Returns concatenated array
        - Dictionaries: Returns dict with concatenated values for each key
        - Tuples: Returns tuple with concatenated values at each position
        - Lists: Returns list with concatenated values at each position
        - Other sequences: Returns same type with concatenated values

    Raises
    ------
    TypeError
        If all elements are not of the same type or if array types are inconsistent.

    Notes
    -----
    This function is particularly useful for:

    - Combining batches of data from different sources
    - Merging predictions from multiple model runs
    - Concatenating results from distributed processing
    - Assembling data from multiple time steps or sequences

    The function automatically detects the array type (PyTorch tensor or NumPy array)
    and uses the appropriate concatenation function (`torch.cat` or `np.concatenate`).

    For nested structures, all elements at each level must have the same keys/indices
    and the same array types.

    Examples
    --------
    Concatenate simple arrays:

    >>> import torch
    >>> import numpy as np
    >>> from transformertf.utils.ops import concatenate
    >>>
    >>> # PyTorch tensors
    >>> tensors = [torch.randn(10, 64), torch.randn(15, 64), torch.randn(20, 64)]
    >>> result = concatenate(tensors)
    >>> print(result.shape)
    torch.Size([45, 64])
    >>>
    >>> # NumPy arrays
    >>> arrays = [np.random.randn(10, 32), np.random.randn(5, 32)]
    >>> result = concatenate(arrays)
    >>> print(result.shape)
    (15, 32)

    Concatenate dictionaries:

    >>> batch1 = {"input": torch.randn(16, 128), "target": torch.randn(16, 10)}
    >>> batch2 = {"input": torch.randn(16, 128), "target": torch.randn(16, 10)}
    >>> combined = concatenate([batch1, batch2])
    >>> print(combined["input"].shape)
    torch.Size([32, 128])

    Concatenate tuples:

    >>> tuple1 = (torch.randn(8, 64), torch.randn(8, 32))
    >>> tuple2 = (torch.randn(12, 64), torch.randn(12, 32))
    >>> combined = concatenate([tuple1, tuple2])
    >>> print(combined[0].shape, combined[1].shape)
    torch.Size([20, 64]) torch.Size([20, 32])

    Concatenate model outputs:

    >>> outputs = []
    >>> for batch in dataloader:
    ...     with torch.no_grad():
    ...         output = model(batch)
    ...         outputs.append(output)
    >>>
    >>> # Combine all outputs
    >>> all_outputs = concatenate(outputs)
    >>> print(all_outputs.shape)
    torch.Size([total_samples, output_dim])

    Concatenate nested structures:

    >>> results = [
    ...     {"predictions": torch.randn(10, 5), "scores": torch.randn(10)},
    ...     {"predictions": torch.randn(15, 5), "scores": torch.randn(15)},
    ...     {"predictions": torch.randn(20, 5), "scores": torch.randn(20)}
    ... ]
    >>> combined = concatenate(results)
    >>> print(combined["predictions"].shape)
    torch.Size([45, 5])

    Use in training loop:

    >>> all_predictions = []
    >>> all_targets = []
    >>>
    >>> for batch in dataloader:
    ...     predictions = model(batch["input"])
    ...     all_predictions.append(predictions)
    ...     all_targets.append(batch["target"])
    >>>
    >>> # Combine all predictions and targets
    >>> final_predictions = concatenate(all_predictions)
    >>> final_targets = concatenate(all_targets)
    >>>
    >>> # Calculate overall metrics
    >>> accuracy = calculate_accuracy(final_predictions, final_targets)

    See Also
    --------
    torch.cat : PyTorch's concatenation function
    numpy.concatenate : NumPy's concatenation function
    truncate : Truncate data to specific length
    """
    if isinstance(value[0], torch.Tensor):
        value = typing.cast(
            typing.Sequence[torch.Tensor],
            value,
        )
        return torch.cat(list(value))
    if isinstance(value[0], np.ndarray):
        return np.concatenate(list(value))

    if isinstance(value[0], dict):
        if not all(isinstance(x, dict) for x in value):
            msg = "All arguments must be of the same type."
            raise TypeError(msg)

        value = typing.cast(
            typing.Sequence[dict[str, arr_t]],
            value,
        )
        arr_type = type(next(iter(value[0].values())))

        if not all(all(isinstance(x, arr_type) for x in d.values()) for d in value):
            msg = "All arguments must be of the same type."
            raise TypeError(msg)

        return type(value[0])({k: concatenate([x[k] for x in value]) for k in value[0]})

    type_ = type(value[0])
    if not all(isinstance(x, type_) for x in value):
        msg = "All arguments must be of the same type."
        raise TypeError(msg)

    if isinstance(value[0], tuple) and all(isinstance(x, tuple) for x in value):
        value = typing.cast(
            typing.Sequence[tuple[arr_t, ...]],
            value,
        )
        return type_(  # type: ignore[call-arg]
            [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
        )
    if isinstance(value[0], list) and all(isinstance(x, list) for x in value):
        if all(isinstance(x, tuple) for x in value[0]):
            value = typing.cast(
                typing.Sequence[tuple[arr_t, ...]],
                value,
            )
            return type_(  # type: ignore[call-arg]
                [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
            )
        value = typing.cast(
            typing.Sequence[list[arr_t]],
            value,
        )
        return type_(  # type: ignore[call-arg]
            [concatenate([x[i] for x in value]) for i in range(len(value[0]))]
        )
    if isinstance(value[0], typing.Sequence):
        value = typing.cast(
            typing.Sequence[typing.Sequence[arr_t]],
            value,
        )
        return type(value[0])(*[
            concatenate([x[i] for x in value]) for i in range(len(value[0]))
        ])

    msg = f"Unknown type {type(value[0])}."
    raise TypeError(msg)


def slice(data: M_co, s: builtins.slice) -> M_co:  # noqa: A001
    """
    Apply slice operation to tensor data recursively.

    This function recursively traverses nested data structures and applies
    the specified slice operation to all tensors. It's useful for extracting
    specific ranges from tensors in complex data structures.

    Parameters
    ----------
    data : M_co
        The data structure to slice. Can be a tensor, tuple, list, sequence,
        or any nested combination of these types containing tensors.
    s : builtins.slice
        The slice object to apply to each tensor. Created using standard
        Python slice syntax, e.g., `slice(start, stop, step)`.

    Returns
    -------
    M_co
        The same data structure with the slice operation applied to all tensors.
        The structure is preserved, but tensors will be sliced according to
        the specified slice object.

    Notes
    -----
    This function is useful for:

    - Extracting specific ranges from sequences
    - Selecting subsets of data for processing
    - Implementing sliding window operations
    - Creating overlapping segments from continuous data

    The slice operation is applied to the first dimension (dimension 0) of
    each tensor. For multi-dimensional slicing, use direct tensor indexing.

    Examples
    --------
    Extract middle portion of tensors:

    >>> import torch
    >>> from transformertf.utils.ops import slice
    >>>
    >>> x = torch.randn(100, 64)
    >>> middle_slice = slice(x, slice(25, 75))
    >>> print(middle_slice.shape)
    torch.Size([50, 64])

    Slice batch data:

    >>> batch = {
    ...     "input": torch.randn(32, 128),
    ...     "target": torch.randn(32, 10)
    ... }
    >>> # Get first 16 samples
    >>> first_half = slice(batch, slice(0, 16))
    >>> print(first_half["input"].shape)
    torch.Size([16, 128])

    Extract with step:

    >>> sequence = torch.randn(1000, 64)
    >>> # Every 10th element
    >>> downsampled = slice(sequence, slice(0, None, 10))
    >>> print(downsampled.shape)
    torch.Size([100, 64])

    Sliding window processing:

    >>> data = torch.randn(1000, 32)
    >>> window_size = 100
    >>> step = 50
    >>>
    >>> windows = []
    >>> for i in range(0, len(data) - window_size, step):
    ...     window = slice(data, slice(i, i + window_size))
    ...     windows.append(window)
    >>>
    >>> print(len(windows), windows[0].shape)
    18 torch.Size([100, 32])

    Extract last N elements:

    >>> last_10 = slice(data, slice(-10, None))
    >>> print(last_10.shape)
    torch.Size([10, 32])

    Use with nested structures:

    >>> nested_data = {
    ...     "sequences": [
    ...         torch.randn(200, 64),
    ...         torch.randn(150, 64)
    ...     ],
    ...     "metadata": torch.randn(200, 8)
    ... }
    >>> # Extract first 100 elements from all tensors
    >>> subset = slice(nested_data, slice(0, 100))
    >>> print(subset["sequences"][0].shape)
    torch.Size([100, 64])

    See Also
    --------
    truncate : Truncate to specific length
    concatenate : Concatenate data structures
    """
    return _op_T(data, lambda x: x[s])  # type: ignore[return-value]


def isnamedtupleinstance(x: typing.Any) -> bool:
    """
    Determine if an object is a namedtuple instance.

    This function checks whether the given object is an instance of a namedtuple
    by examining its type hierarchy and attributes. It's used internally by
    other utility functions to handle namedtuples appropriately.

    Parameters
    ----------
    x : typing.Any
        The object to check for namedtuple instance.

    Returns
    -------
    bool
        True if the object is a namedtuple instance, False otherwise.

    Notes
    -----
    This function performs the following checks:

    1. Examines the type hierarchy to find if `tuple` is a superclass
    2. Checks for the presence of a `_fields` attribute
    3. Verifies that `_fields` is a tuple of strings

    These checks ensure that the object follows the namedtuple pattern
    and can be handled appropriately by other utility functions.

    Examples
    --------
    Check regular tuple:

    >>> from transformertf.utils.ops import isnamedtupleinstance
    >>> regular_tuple = (1, 2, 3)
    >>> print(isnamedtupleinstance(regular_tuple))
    False

    Check namedtuple:

    >>> from collections import namedtuple
    >>> Point = namedtuple('Point', ['x', 'y'])
    >>> point = Point(1, 2)
    >>> print(isnamedtupleinstance(point))
    True

    Check other types:

    >>> print(isnamedtupleinstance([1, 2, 3]))  # List
    False
    >>> print(isnamedtupleinstance({'a': 1}))   # Dict
    False

    Use in data processing:

    >>> def process_data(data):
    ...     if isnamedtupleinstance(data):
    ...         # Handle namedtuple specially
    ...         return type(data)(*[process_item(item) for item in data])
    ...     else:
    ...         # Handle other types
    ...         return process_other(data)

    See Also
    --------
    collections.namedtuple : Python's namedtuple factory
    _op_T : Internal function that uses this for type checking
    """
    t = type(x)
    superclasses: set[type] = set()

    def find_superclasses(t: type) -> None:
        nonlocal superclasses
        bases = list(t.__bases__)
        for b in bases:
            find_superclasses(b)

        superclasses |= set(bases)

    find_superclasses(t)
    if tuple not in superclasses:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(isinstance(n, str) for n in f)
