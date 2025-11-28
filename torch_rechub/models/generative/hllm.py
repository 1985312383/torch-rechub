"""HLLM: Hierarchical Large Language Model for Recommendation.

This module implements the HLLM model from ByteDance with support for both
lightweight and full LLM architectures.
Reference: https://github.com/bytedance/HLLM (arXiv:2409.12740)

Supported modes:
1. **Lightweight Mode** (default): Pre-computed item embeddings + lightweight Transformer blocks
   - Memory: ~100MB - 1GB depending on vocab size
   - Suitable for: Learning, small-scale experiments, resource-constrained environments

2. **Item LLM Mode**: Online item embedding extraction using TinyLlama/Baichuan2
   - Memory: +2GB (TinyLlama-1.1B) or +14GB (Baichuan2-7B)
   - Suitable for: Dynamic item catalogues, online feature extraction

3. **User LLM Mode**: Full LLM as User encoder
   - Memory: +2GB (TinyLlama-1.1B) or +14GB (Baichuan2-7B)
   - Suitable for: Maximum model capacity, paper reproduction

4. **Full HLLM Mode**: Both Item LLM and User LLM
   - Memory: ~4GB (TinyLlama x2) or ~30GB (Baichuan2 x2)
   - Suitable for: Industrial deployment, paper reproduction

Memory estimates (FP16):
- TinyLlama-1.1B: ~2.2GB per model
- Baichuan2-7B: ~14GB per model
- 8-bit quantization: ~50% memory reduction
- 4-bit quantization: ~75% memory reduction

This implementation aims to match the official architecture details:
- SiLU activation (matches Llama/Baichuan)
- RMSNorm (matches Llama/Baichuan)
- Optional SwiGLU FFN variant
- [ITEM] special token for item embedding extraction
"""

import math
import warnings
from typing import Optional, Union, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_rechub.utils.hstu_utils import RelPosBias

# Optional imports for LLM support
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None


def _check_transformers_available():
    """Check if transformers library is available."""
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers library is required for LLM support. "
            "Install it with: pip install transformers>=4.30.0"
        )


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    This is the normalization used in Llama and other modern LLMs.
    Reference: https://arxiv.org/abs/1910.07467

    Args:
        d_model (int): Hidden dimension.
        eps (float): Epsilon for numerical stability. Default: 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (..., d_model).

        Returns:
            Tensor: Normalized output of same shape.
        """
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return x / rms * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function used in Llama and modern LLMs.

    SwiGLU(x) = Swish(W1 @ x) * (W2 @ x)
    where Swish(x) = x * sigmoid(x) = SiLU(x)

    Reference: https://arxiv.org/abs/2002.05202

    Args:
        d_model (int): Input dimension.
        ffn_hidden (int): Hidden dimension (typically 4 * d_model or 8/3 * d_model).
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w2 = nn.Linear(d_model, ffn_hidden, bias=False)
        self.w3 = nn.Linear(ffn_hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (..., d_model).

        Returns:
            Tensor: Output of shape (..., d_model).
        """
        # SwiGLU: SiLU(W1 @ x) * (W2 @ x)
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class ItemLLM(nn.Module):
    """Item LLM for online item embedding extraction.

    This module wraps a pre-trained LLM (TinyLlama or Baichuan2) to extract
    item embeddings using the [ITEM] special token, matching the official
    HLLM implementation.

    The item embedding is extracted from the hidden state at the [ITEM] token
    position in the last layer of the LLM.

    Memory requirements (FP16):
    - TinyLlama-1.1B: ~2.2GB
    - Baichuan2-7B: ~14GB
    - With 8-bit quantization: ~50% reduction
    - With 4-bit quantization: ~75% reduction

    Args:
        model_path (str): Path or HuggingFace model ID for the LLM.
            Supported: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0', 'baichuan-inc/Baichuan2-7B-Base'
        item_token (str): Special token for item embedding. Default: '[ITEM]'.
        freeze (bool): Whether to freeze LLM parameters. Default: True.
        load_in_8bit (bool): Whether to use 8-bit quantization. Default: False.
        load_in_4bit (bool): Whether to use 4-bit quantization. Default: False.
        device_map (str): Device mapping strategy. Default: 'auto'.
        torch_dtype: Model dtype. Default: torch.float16.
        trust_remote_code (bool): Whether to trust remote code. Default: True.

    Example:
        >>> item_llm = ItemLLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0', freeze=True)
        >>> item_texts = ['Movie: The Matrix', 'Movie: Inception']
        >>> embeddings = item_llm(item_texts)  # (2, hidden_size)
    """

    # Recommended model configurations
    SUPPORTED_MODELS = {
        'tinyllama': {
            'model_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'hidden_size': 2048,
            'memory_fp16': '2.2GB',
            'memory_8bit': '1.1GB',
            'memory_4bit': '0.6GB',
        },
        'baichuan2-7b': {
            'model_id': 'baichuan-inc/Baichuan2-7B-Base',
            'hidden_size': 4096,
            'memory_fp16': '14GB',
            'memory_8bit': '7GB',
            'memory_4bit': '3.5GB',
        },
    }

    def __init__(
        self,
        model_path: str,
        item_token: str = '[ITEM]',
        freeze: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = 'auto',
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        _check_transformers_available()

        self.model_path = model_path
        self.item_token = item_token
        self.freeze = freeze
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Configure quantization
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install it with: pip install bitsandbytes>=0.39.0"
                )
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        # Add special token if not exists
        if item_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': [item_token]})
        self.item_token_id = self.tokenizer.convert_tokens_to_ids(item_token)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )

        # Resize token embeddings if new tokens were added
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Get hidden size
        self.hidden_size = self.model.config.hidden_size

        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(
        self,
        item_texts: List[str],
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Extract item embeddings from text descriptions.

        Args:
            item_texts (List[str]): List of item text descriptions.
                Each text should NOT include the [ITEM] token - it will be added.
            return_dict (bool): Whether to return a dict with additional info.

        Returns:
            Tensor: Item embeddings of shape (batch_size, hidden_size).
            Or dict with 'embeddings' and 'hidden_states' if return_dict=True.
        """
        # Add [ITEM] token to each text
        texts_with_item = [f"{text} {self.item_token}" for text in item_texts]

        # Tokenize
        inputs = self.tokenizer(
            texts_with_item,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )

        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Get hidden states from last layer
        last_hidden_states = outputs.hidden_states[-1]  # (B, L, H)

        # Find [ITEM] token positions and extract embeddings
        item_positions = (inputs['input_ids'] == self.item_token_id).nonzero(as_tuple=True)

        # Extract embeddings at [ITEM] positions
        embeddings = last_hidden_states[item_positions[0], item_positions[1]]  # (B, H)

        if return_dict:
            return {
                'embeddings': embeddings,
                'hidden_states': last_hidden_states,
            }
        return embeddings

    @torch.no_grad()
    def encode_items(
        self,
        item_texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Encode multiple items in batches.

        Args:
            item_texts (List[str]): List of item text descriptions.
            batch_size (int): Batch size for encoding. Default: 32.
            show_progress (bool): Whether to show progress bar. Default: True.

        Returns:
            Tensor: Item embeddings of shape (num_items, hidden_size).
        """
        all_embeddings = []

        iterator = range(0, len(item_texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc='Encoding items')
            except ImportError:
                pass

        for i in iterator:
            batch_texts = item_texts[i:i + batch_size]
            embeddings = self.forward(batch_texts)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


class UserLLM(nn.Module):
    """User LLM for sequence modeling using a pre-trained LLM.

    This module wraps a pre-trained LLM (TinyLlama or Baichuan2) for user
    sequence modeling, matching the official HLLM implementation.

    The User LLM takes item embeddings as input and outputs user representations.
    A projection layer is used to map item embeddings to the LLM's input dimension
    if dimensions don't match.

    Memory requirements (FP16):
    - TinyLlama-1.1B: ~2.2GB
    - Baichuan2-7B: ~14GB
    - With gradient checkpointing: ~30% reduction during training

    Args:
        model_path (str): Path or HuggingFace model ID for the LLM.
        input_dim (int): Dimension of input item embeddings.
        freeze (bool): Whether to freeze LLM parameters. Default: False.
        load_in_8bit (bool): Whether to use 8-bit quantization. Default: False.
        load_in_4bit (bool): Whether to use 4-bit quantization. Default: False.
        use_gradient_checkpointing (bool): Enable gradient checkpointing. Default: False.
        device_map (str): Device mapping strategy. Default: 'auto'.
        torch_dtype: Model dtype. Default: torch.float16.
        trust_remote_code (bool): Whether to trust remote code. Default: True.

    Example:
        >>> user_llm = UserLLM('TinyLlama/TinyLlama-1.1B-Chat-v1.0', input_dim=2048)
        >>> item_embeddings = torch.randn(32, 50, 2048)  # (batch, seq_len, dim)
        >>> user_repr = user_llm(item_embeddings)  # (32, 50, hidden_size)
    """

    def __init__(
        self,
        model_path: str,
        input_dim: int,
        freeze: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_gradient_checkpointing: bool = False,
        device_map: str = 'auto',
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        _check_transformers_available()

        self.model_path = model_path
        self.input_dim = input_dim
        self.freeze = freeze
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Configure quantization
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            if BitsAndBytesConfig is None:
                raise ImportError(
                    "bitsandbytes is required for quantization. "
                    "Install it with: pip install bitsandbytes>=0.39.0"
                )
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype if load_in_4bit else None,
            )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )

        # Get hidden size
        self.hidden_size = self.model.config.hidden_size

        # Input projection if dimensions don't match
        if input_dim != self.hidden_size:
            self.input_projection = nn.Linear(input_dim, self.hidden_size)
        else:
            self.input_projection = nn.Identity()

        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Freeze parameters if requested
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(
        self,
        item_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the User LLM.

        Args:
            item_embeddings (Tensor): Item embeddings of shape (B, L, D).
            attention_mask (Tensor, optional): Attention mask of shape (B, L).

        Returns:
            Tensor: User representations of shape (B, L, hidden_size).
        """
        # Project input if needed
        inputs_embeds = self.input_projection(item_embeddings)

        # Create causal attention mask if not provided
        batch_size, seq_len, _ = inputs_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=inputs_embeds.device)

        # Forward pass through LLM
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Return last hidden states
        return outputs.hidden_states[-1]

    def get_output_dim(self) -> int:
        """Get the output dimension of the User LLM."""
        return self.hidden_size


class HLLMTransformerBlock(nn.Module):
    """Single HLLM Transformer block with self-attention and FFN.

    This block is designed to match the architecture of modern LLMs (Llama/Baichuan)
    used in the official HLLM implementation.

    Key architecture choices (matching official implementation):
    - RMSNorm instead of LayerNorm (used in Llama/Baichuan)
    - SiLU activation instead of ReLU (used in Llama/Baichuan)
    - Optional SwiGLU FFN variant (used in Llama)
    - Pre-norm architecture

    Args:
        d_model (int): Hidden dimension. Default: 512.
        n_heads (int): Number of attention heads. Default: 8.
        dropout (float): Dropout rate. Default: 0.1.
        use_swiglu (bool): Whether to use SwiGLU FFN. Default: False.
            Set to True to match Llama architecture more closely.
        norm_type (str): Normalization type ('rmsnorm' or 'layernorm'). Default: 'rmsnorm'.
            'rmsnorm' matches Llama/Baichuan, 'layernorm' for backward compatibility.
        ffn_multiplier (float): FFN hidden dimension multiplier. Default: 4.0.
            For SwiGLU, Llama uses 8/3 ≈ 2.67, but we keep 4.0 for compatibility.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        norm_type: str = 'rmsnorm',
        ffn_multiplier: float = 4.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_swiglu = use_swiglu
        self.norm_type = norm_type

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Multi-head self-attention
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        # Feed-forward network
        ffn_hidden = int(ffn_multiplier * d_model)

        if use_swiglu:
            # SwiGLU FFN (matches Llama architecture)
            self.ffn = SwiGLU(d_model, ffn_hidden, dropout)
        else:
            # Standard FFN with SiLU activation (matches Llama/Baichuan activation)
            # Changed from ReLU to SiLU to match official implementation
            self.ffn = nn.Sequential(
                nn.Linear(d_model, ffn_hidden),
                nn.SiLU(),  # Changed from nn.ReLU() to match Llama/Baichuan
                nn.Dropout(dropout),
                nn.Linear(ffn_hidden, d_model),
                nn.Dropout(dropout)
            )

        # Normalization layers
        # RMSNorm is used in Llama/Baichuan, LayerNorm available for backward compatibility
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        elif norm_type == 'layernorm':
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}. Use 'rmsnorm' or 'layernorm'.")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, rel_pos_bias=None):
        """Forward pass.
        
        Args:
            x (Tensor): Input of shape (B, L, D).
            rel_pos_bias (Tensor, optional): Relative position bias.
            
        Returns:
            Tensor: Output of shape (B, L, D).
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention with residual
        residual = x
        x = self.norm1(x)

        Q = self.W_Q(x)  # (B, L, D)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Add relative position bias if provided
        if rel_pos_bias is not None:
            scores = scores + rel_pos_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.W_O(attn_output)
        attn_output = self.dropout(attn_output)

        x = residual + attn_output

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class HLLMModel(nn.Module):
    """HLLM: Hierarchical Large Language Model for Recommendation.

    This implementation supports multiple modes:

    1. **Lightweight Mode** (default): Pre-computed embeddings + lightweight Transformer
       - Set: item_embeddings=<tensor>, item_llm_path=None, user_llm_path=None
       - Memory: ~100MB - 1GB

    2. **Item LLM Mode**: Online item embedding + lightweight Transformer
       - Set: item_llm_path='TinyLlama/...', user_llm_path=None
       - Memory: +2GB (TinyLlama) or +14GB (Baichuan2)

    3. **User LLM Mode**: Pre-computed embeddings + full User LLM
       - Set: item_embeddings=<tensor>, user_llm_path='TinyLlama/...'
       - Memory: +2GB (TinyLlama) or +14GB (Baichuan2)

    4. **Full HLLM Mode**: Both Item LLM and User LLM (matches official implementation)
       - Set: item_llm_path='TinyLlama/...', user_llm_path='TinyLlama/...'
       - Memory: ~4GB (TinyLlama x2) or ~30GB (Baichuan2 x2)

    Memory estimates (FP16, for reference):
    +-------------------+--------+---------+---------+
    | Mode              | 16GB   | 24GB    | 32GB+   |
    +-------------------+--------+---------+---------+
    | Lightweight       | ✓      | ✓       | ✓       |
    | TinyLlama Item    | ✓      | ✓       | ✓       |
    | TinyLlama User    | ✓      | ✓       | ✓       |
    | TinyLlama x2      | ✓*     | ✓       | ✓       |
    | Baichuan2 Item    | ✓*     | ✓       | ✓       |
    | Baichuan2 User    | ✓*     | ✓       | ✓       |
    | Baichuan2 x2      | ✗      | ✗       | ✓       |
    +-------------------+--------+---------+---------+
    * With 8-bit quantization

    Reference: https://github.com/bytedance/HLLM (arXiv:2409.12740)

    Args:
        item_embeddings: Pre-computed item embeddings (Tensor, path, or None).
            If None, item_llm_path must be provided.
        vocab_size (int): Number of items. Required.
        d_model (int): Hidden dimension for lightweight mode. Default: 512.
        n_heads (int): Number of attention heads. Default: 8.
        n_layers (int): Number of transformer blocks. Default: 4.
        max_seq_len (int): Maximum sequence length. Default: 256.
        dropout (float): Dropout rate. Default: 0.1.
        use_rel_pos_bias (bool): Whether to use relative position bias. Default: True.
        use_time_embedding (bool): Whether to use time embeddings. Default: True.
        num_time_buckets (int): Number of time buckets. Default: 2048.
        time_bucket_fn (str): Time bucketization function. Default: 'sqrt'.
        temperature (float): Temperature for scoring. Default: 1.0.
        use_swiglu (bool): Whether to use SwiGLU FFN. Default: False.
        norm_type (str): Normalization type. Default: 'rmsnorm'.
        ffn_multiplier (float): FFN hidden dimension multiplier. Default: 4.0.

        # Item LLM parameters
        item_llm_path (str, optional): Path to Item LLM model. Default: None.
        item_texts (List[str], optional): Item text descriptions for online extraction.
        freeze_item_llm (bool): Whether to freeze Item LLM. Default: True.
        item_llm_8bit (bool): Use 8-bit quantization for Item LLM. Default: False.
        item_llm_4bit (bool): Use 4-bit quantization for Item LLM. Default: False.

        # User LLM parameters
        user_llm_path (str, optional): Path to User LLM model. Default: None.
        freeze_user_llm (bool): Whether to freeze User LLM. Default: False.
        user_llm_8bit (bool): Use 8-bit quantization for User LLM. Default: False.
        user_llm_4bit (bool): Use 4-bit quantization for User LLM. Default: False.
        user_llm_gradient_checkpointing (bool): Enable gradient checkpointing. Default: False.

    Examples:
        >>> # Mode 1: Lightweight (default, backward compatible)
        >>> item_embeddings = torch.randn(1000, 512)
        >>> model = HLLMModel(item_embeddings=item_embeddings, vocab_size=1000)

        >>> # Mode 2: With Item LLM
        >>> model = HLLMModel(
        ...     item_embeddings=None,  # Will use Item LLM
        ...     vocab_size=1000,
        ...     item_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ...     item_texts=['Movie: The Matrix', 'Movie: Inception', ...],
        ... )

        >>> # Mode 3: With User LLM
        >>> model = HLLMModel(
        ...     item_embeddings=item_embeddings,
        ...     vocab_size=1000,
        ...     user_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ... )

        >>> # Mode 4: Full HLLM (matches official implementation)
        >>> model = HLLMModel(
        ...     item_embeddings=None,
        ...     vocab_size=1000,
        ...     item_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ...     item_texts=['Movie: The Matrix', ...],
        ...     user_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ...     freeze_item_llm=True,
        ...     freeze_user_llm=False,
        ... )

        >>> # Memory-optimized configuration for 16GB GPU
        >>> model = HLLMModel(
        ...     item_embeddings=None,
        ...     vocab_size=1000,
        ...     item_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ...     item_texts=[...],
        ...     user_llm_path='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        ...     item_llm_8bit=True,
        ...     user_llm_8bit=True,
        ...     user_llm_gradient_checkpointing=True,
        ... )
    """

    def __init__(
        self,
        item_embeddings: Optional[Union[torch.Tensor, str]] = None,
        vocab_size: int = None,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        use_rel_pos_bias: bool = True,
        use_time_embedding: bool = True,
        num_time_buckets: int = 2048,
        time_bucket_fn: str = 'sqrt',
        temperature: float = 1.0,
        use_swiglu: bool = False,
        norm_type: str = 'rmsnorm',
        ffn_multiplier: float = 4.0,
        # Item LLM parameters
        item_llm_path: Optional[str] = None,
        item_texts: Optional[List[str]] = None,
        freeze_item_llm: bool = True,
        item_llm_8bit: bool = False,
        item_llm_4bit: bool = False,
        # User LLM parameters
        user_llm_path: Optional[str] = None,
        freeze_user_llm: bool = False,
        user_llm_8bit: bool = False,
        user_llm_4bit: bool = False,
        user_llm_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        # Validate inputs
        if item_embeddings is None and item_llm_path is None:
            raise ValueError(
                "Either item_embeddings or item_llm_path must be provided. "
                "Use item_embeddings for lightweight mode, or item_llm_path for full LLM mode."
            )

        if item_llm_path is not None and item_texts is None:
            raise ValueError(
                "item_texts must be provided when using item_llm_path. "
                "This should be a list of text descriptions for all items."
            )

        if vocab_size is None:
            raise ValueError("vocab_size must be provided.")

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_time_embedding = use_time_embedding
        self.num_time_buckets = num_time_buckets
        self.time_bucket_fn = time_bucket_fn
        self.temperature = temperature
        self.use_swiglu = use_swiglu
        self.norm_type = norm_type
        self.ffn_multiplier = ffn_multiplier

        # Mode flags
        self.use_item_llm = item_llm_path is not None
        self.use_user_llm = user_llm_path is not None

        # Initialize Item LLM or load pre-computed embeddings
        if self.use_item_llm:
            self.item_llm = ItemLLM(
                model_path=item_llm_path,
                freeze=freeze_item_llm,
                load_in_8bit=item_llm_8bit,
                load_in_4bit=item_llm_4bit,
            )
            # Extract item embeddings using Item LLM
            print(f"Extracting item embeddings using {item_llm_path}...")
            with torch.no_grad():
                item_embeddings_tensor = self.item_llm.encode_items(item_texts)
            self.register_buffer('item_embeddings', item_embeddings_tensor.float())

            # Update d_model to match Item LLM output
            actual_d_model = self.item_llm.hidden_size
            if d_model != actual_d_model:
                warnings.warn(
                    f"d_model ({d_model}) does not match Item LLM hidden size ({actual_d_model}). "
                    f"Using {actual_d_model} instead."
                )
                self.d_model = actual_d_model
                d_model = actual_d_model
        else:
            self.item_llm = None
            # Load pre-computed item embeddings
            if isinstance(item_embeddings, str):
                item_embeddings = torch.load(item_embeddings)
            self.register_buffer('item_embeddings', item_embeddings.float())

            # Verify d_model matches embedding dimension
            if item_embeddings.shape[1] != d_model:
                warnings.warn(
                    f"d_model ({d_model}) does not match item_embeddings dimension ({item_embeddings.shape[1]}). "
                    f"Using {item_embeddings.shape[1]} instead."
                )
                self.d_model = item_embeddings.shape[1]
                d_model = item_embeddings.shape[1]

        # Initialize User LLM or lightweight Transformer
        if self.use_user_llm:
            self.user_llm = UserLLM(
                model_path=user_llm_path,
                input_dim=d_model,
                freeze=freeze_user_llm,
                load_in_8bit=user_llm_8bit,
                load_in_4bit=user_llm_4bit,
                use_gradient_checkpointing=user_llm_gradient_checkpointing,
            )
            self.transformer_blocks = None
            self.rel_pos_bias = None

            # Output projection if User LLM hidden size differs from item embedding dim
            user_output_dim = self.user_llm.get_output_dim()
            if user_output_dim != d_model:
                self.output_projection = nn.Linear(user_output_dim, d_model)
            else:
                self.output_projection = nn.Identity()
        else:
            self.user_llm = None
            self.output_projection = nn.Identity()

            # Lightweight Transformer blocks
            self.transformer_blocks = nn.ModuleList([
                HLLMTransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_swiglu=use_swiglu,
                    norm_type=norm_type,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(n_layers)
            ])

            # Relative position bias
            self.use_rel_pos_bias = use_rel_pos_bias
            if use_rel_pos_bias:
                self.rel_pos_bias = RelPosBias(n_heads, max_seq_len)
            else:
                self.rel_pos_bias = None

        # Positional embedding (for lightweight mode only)
        if not self.use_user_llm:
            self.position_embedding = nn.Embedding(max_seq_len, d_model)
        else:
            self.position_embedding = None

        # Time embedding
        if use_time_embedding:
            self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)
        else:
            self.time_embedding = None

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights for lightweight components
        if not self.use_user_llm:
            self._init_weights()

        # Log configuration
        self._log_config()

    def _log_config(self):
        """Log model configuration."""
        mode = []
        if self.use_item_llm:
            mode.append("Item LLM")
        else:
            mode.append("Pre-computed embeddings")
        if self.use_user_llm:
            mode.append("User LLM")
        else:
            mode.append("Lightweight Transformer")

        print(f"HLLM initialized with: {' + '.join(mode)}")
        print(f"  - vocab_size: {self.vocab_size}")
        print(f"  - d_model: {self.d_model}")
        if not self.use_user_llm:
            print(f"  - n_layers: {self.n_layers}")
            print(f"  - n_heads: {self.n_heads}")

    def _init_weights(self):
        """Initialize model parameters."""
        for name, param in self.named_parameters():
            if 'item_llm' in name or 'user_llm' in name:
                continue  # Skip LLM parameters
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def _time_diff_to_bucket(self, time_diffs):
        """Map time differences to bucket indices."""
        time_diffs = time_diffs.float() / 60.0  # seconds to minutes
        time_diffs = torch.clamp(time_diffs, min=1e-6)

        if self.time_bucket_fn == 'sqrt':
            buckets = torch.sqrt(time_diffs).long()
        elif self.time_bucket_fn == 'log':
            buckets = torch.log(time_diffs).long()
        else:
            raise ValueError(f"Unsupported time_bucket_fn: {self.time_bucket_fn}")

        buckets = torch.clamp(buckets, min=0, max=self.num_time_buckets - 1)
        return buckets

    def forward(
        self,
        seq_tokens: torch.Tensor,
        time_diffs: Optional[torch.Tensor] = None,
        item_texts: Optional[List[List[str]]] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            seq_tokens (Tensor): Item token IDs of shape (B, L).
                Used to look up pre-computed embeddings.
            time_diffs (Tensor, optional): Time differences in seconds of shape (B, L).
            item_texts (List[List[str]], optional): Batch of item text sequences.
                Only used when use_item_llm=True and you want online extraction.
                If None, uses pre-extracted embeddings from initialization.

        Returns:
            Tensor: Logits of shape (B, L, vocab_size).
        """
        batch_size, seq_len = seq_tokens.shape

        # Get item embeddings
        if item_texts is not None and self.item_llm is not None:
            # Online item embedding extraction (expensive, use sparingly)
            flat_texts = [text for seq in item_texts for text in seq]
            flat_embeddings = self.item_llm(flat_texts)
            item_emb = flat_embeddings.view(batch_size, seq_len, -1)
        else:
            # Use pre-computed or pre-extracted embeddings
            item_emb = self.item_embeddings[seq_tokens]  # (B, L, D)

        # Add time embedding if enabled
        if self.use_time_embedding and self.time_embedding is not None:
            if time_diffs is None:
                time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long, device=seq_tokens.device)
            time_buckets = self._time_diff_to_bucket(time_diffs)
            time_emb = self.time_embedding(time_buckets)  # (B, L, D)
            item_emb = item_emb + time_emb

        # Process through User model
        if self.use_user_llm:
            # Use User LLM
            x = self.user_llm(item_emb)
            x = self.output_projection(x)
        else:
            # Use lightweight Transformer
            # Add positional embedding
            positions = torch.arange(seq_len, dtype=torch.long, device=seq_tokens.device)
            pos_emb = self.position_embedding(positions)  # (L, D)
            embeddings = item_emb + pos_emb.unsqueeze(0)  # (B, L, D)

            embeddings = self.dropout(embeddings)

            # Get relative position bias
            rel_pos_bias = None
            if self.use_rel_pos_bias and self.rel_pos_bias is not None:
                rel_pos_bias = self.rel_pos_bias(seq_len)

            # Pass through transformer blocks
            x = embeddings
            for block in self.transformer_blocks:
                x = block(x, rel_pos_bias=rel_pos_bias)

        # Scoring head: compute dot product with item embeddings
        # x: (B, L, D), item_embeddings: (V, D)
        logits = torch.matmul(x, self.item_embeddings.t()) / self.temperature  # (B, L, V)

        return logits

    def update_item_embeddings(
        self,
        item_texts: List[str],
        batch_size: int = 32,
    ):
        """Update item embeddings using the Item LLM.

        This is useful when items are added or their descriptions change.
        Only available when use_item_llm=True.

        Args:
            item_texts (List[str]): Updated item text descriptions.
            batch_size (int): Batch size for encoding. Default: 32.
        """
        if self.item_llm is None:
            raise RuntimeError(
                "update_item_embeddings is only available when item_llm_path is provided. "
                "For pre-computed embeddings, create a new model instance."
            )

        print(f"Updating item embeddings for {len(item_texts)} items...")
        with torch.no_grad():
            new_embeddings = self.item_llm.encode_items(item_texts, batch_size=batch_size)

        # Update buffer
        self.item_embeddings = new_embeddings.float().to(self.item_embeddings.device)
        self.vocab_size = len(item_texts)
        print(f"Item embeddings updated. New vocab_size: {self.vocab_size}")

    def get_item_embeddings(self) -> torch.Tensor:
        """Get current item embeddings.

        Returns:
            Tensor: Item embeddings of shape (vocab_size, d_model).
        """
        return self.item_embeddings.clone()

    def save_item_embeddings(self, path: str):
        """Save item embeddings to a file.

        Args:
            path (str): Path to save the embeddings.
        """
        torch.save(self.item_embeddings.cpu(), path)
        print(f"Item embeddings saved to {path}")
