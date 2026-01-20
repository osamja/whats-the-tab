"""Convert JAX/T5X MT3 checkpoint to PyTorch format.

This script converts the zarr-format JAX checkpoint to a PyTorch state_dict.
"""

import os
import sys
import zarr
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
import json


def load_zarr_array(zarr_path: str) -> np.ndarray:
    """Load a zarr array from disk.

    Args:
        zarr_path: Path to zarr array directory

    Returns:
        Numpy array with the data
    """
    # For zarr 3.x, we can open directly from the path
    try:
        array = zarr.open(zarr_path, mode='r')
        return np.array(array)
    except Exception:
        # Fallback: try older API
        try:
            from zarr import storage
            store = storage.DirectoryStore(zarr_path)
            array = zarr.open_array(store, mode='r')
            return np.array(array)
        except Exception:
            # Last resort: read the raw data directly
            import json
            # Read .zarray metadata
            with open(os.path.join(zarr_path, '.zarray'), 'r') as f:
                metadata = json.load(f)

            # Read the data file
            data_file = os.path.join(zarr_path, '0.0')
            dtype = np.dtype(metadata['dtype'])
            shape = tuple(metadata['shape'])

            # Read raw data
            data = np.fromfile(data_file, dtype=dtype)
            return data.reshape(shape)


def map_jax_to_pytorch_name(jax_name: str) -> Tuple[str, bool]:
    """Map JAX parameter name to PyTorch parameter name.

    Args:
        jax_name: JAX parameter name (e.g., 'target.encoder.layers_0.attention.query.kernel')

    Returns:
        Tuple of (pytorch_name, needs_transpose)
    """
    # Remove 'target.' prefix
    if jax_name.startswith('target.'):
        jax_name = jax_name[7:]

    needs_transpose = False

    # Encoder continuous inputs projection
    if jax_name == 'encoder.continuous_inputs_projection.kernel':
        return 'encoder.input_projection.weight', True

    # Token embeddings
    if jax_name == 'decoder.token_embedder.embedding':
        return 'decoder.token_embeddings.weight', False

    # Logits dense (output projection)
    if jax_name == 'decoder.logits_dense.kernel':
        return 'decoder.output_projection.weight', True

    # Encoder layers
    if jax_name.startswith('encoder.layers_'):
        # Extract layer number
        parts = jax_name.split('.')
        layer_num = parts[1].split('_')[1]

        if 'attention' in jax_name:
            # encoder.layers_0.attention.query.kernel -> encoder.layers.0.self_attn.q_proj.weight
            if 'query.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.self_attn.q_proj.weight', True
            elif 'key.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.self_attn.k_proj.weight', True
            elif 'value.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.self_attn.v_proj.weight', True
            elif 'out.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.self_attn.out_proj.weight', True

        elif 'mlp' in jax_name:
            # encoder.layers_0.mlp.wi_0.kernel -> encoder.layers.0.mlp.wi_0.weight
            if 'wi_0.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.mlp.wi_0.weight', True
            elif 'wi_1.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.mlp.wi_1.weight', True
            elif 'wo.kernel' in jax_name:
                return f'encoder.layers.{layer_num}.mlp.wo.weight', True

    # Decoder layers
    if jax_name.startswith('decoder.layers_'):
        parts = jax_name.split('.')
        layer_num = parts[1].split('_')[1]

        if 'self_attention' in jax_name:
            # decoder.layers_0.self_attention.query.kernel -> decoder.layers.0.self_attn.q_proj.weight
            if 'query.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.self_attn.q_proj.weight', True
            elif 'key.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.self_attn.k_proj.weight', True
            elif 'value.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.self_attn.v_proj.weight', True
            elif 'out.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.self_attn.out_proj.weight', True

        elif 'encoder_decoder_attention' in jax_name:
            # decoder.layers_0.encoder_decoder_attention.query.kernel -> decoder.layers.0.cross_attn.q_proj.weight
            if 'query.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.cross_attn.q_proj.weight', True
            elif 'key.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.cross_attn.k_proj.weight', True
            elif 'value.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.cross_attn.v_proj.weight', True
            elif 'out.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.cross_attn.out_proj.weight', True

        elif 'mlp' in jax_name:
            # decoder.layers_0.mlp.wi_0.kernel -> decoder.layers.0.mlp.wi_0.weight
            if 'wi_0.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.mlp.wi_0.weight', True
            elif 'wi_1.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.mlp.wi_1.weight', True
            elif 'wo.kernel' in jax_name:
                return f'decoder.layers.{layer_num}.mlp.wo.weight', True

    # If we get here, the name wasn't recognized
    raise ValueError(f"Unknown parameter name: {jax_name}")


def convert_checkpoint(
    jax_checkpoint_dir: str,
    output_path: str,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert JAX checkpoint to PyTorch state_dict.

    Args:
        jax_checkpoint_dir: Path to JAX checkpoint directory
        output_path: Path to save PyTorch checkpoint
        verbose: Whether to print progress

    Returns:
        PyTorch state_dict
    """
    checkpoint_path = Path(jax_checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {jax_checkpoint_dir}")

    # Find all parameter directories
    param_dirs = []
    for item in checkpoint_path.iterdir():
        if item.is_dir() and item.name.startswith('target.'):
            param_dirs.append(item.name)

    if verbose:
        print(f"Found {len(param_dirs)} parameters in checkpoint")
        print(f"Converting from: {jax_checkpoint_dir}")
        print(f"Saving to: {output_path}")
        print()

    # Convert each parameter
    state_dict = {}
    conversion_stats = {
        'total': 0,
        'converted': 0,
        'skipped': 0,
        'errors': 0,
    }

    for jax_param_name in sorted(param_dirs):
        conversion_stats['total'] += 1

        try:
            # Load the array
            param_path = checkpoint_path / jax_param_name
            jax_array = load_zarr_array(str(param_path))

            # Map name and determine if transpose is needed
            pytorch_name, needs_transpose = map_jax_to_pytorch_name(jax_param_name)

            # Convert to PyTorch tensor
            pytorch_tensor = torch.from_numpy(jax_array)

            # Transpose if needed (JAX uses [in_features, out_features], PyTorch uses [out_features, in_features])
            if needs_transpose:
                pytorch_tensor = pytorch_tensor.T

            state_dict[pytorch_name] = pytorch_tensor
            conversion_stats['converted'] += 1

            if verbose:
                shape_str = f"{list(jax_array.shape)}"
                if needs_transpose:
                    shape_str += f" -> {list(pytorch_tensor.shape)}"
                print(f"✓ {jax_param_name:70s} -> {pytorch_name:50s} {shape_str}")

        except Exception as e:
            conversion_stats['errors'] += 1
            print(f"✗ {jax_param_name:70s} ERROR: {e}")

    if verbose:
        print()
        print("=" * 100)
        print(f"Conversion Summary:")
        print(f"  Total parameters: {conversion_stats['total']}")
        print(f"  Converted: {conversion_stats['converted']}")
        print(f"  Errors: {conversion_stats['errors']}")
        print("=" * 100)

    # Save the state dict
    torch.save(state_dict, output_path)

    if verbose:
        print(f"\nSaved PyTorch checkpoint to: {output_path}")
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")

    return state_dict


def verify_conversion(
    state_dict: Dict[str, torch.Tensor],
    verbose: bool = True,
) -> bool:
    """Verify the converted checkpoint has expected structure.

    Args:
        state_dict: PyTorch state_dict
        verbose: Whether to print details

    Returns:
        True if verification passes
    """
    if verbose:
        print("\n" + "=" * 100)
        print("Verifying Conversion")
        print("=" * 100)

    # Expected parameter groups
    expected_groups = {
        'encoder.input_projection': 1,
        'encoder.layers': 8,  # 8 encoder layers
        'decoder.token_embeddings': 1,
        'decoder.layers': 8,  # 8 decoder layers
        'decoder.output_projection': 1,
    }

    # Count parameters in each group
    actual_groups = {}
    for key in state_dict.keys():
        for group in expected_groups.keys():
            if key.startswith(group):
                if group not in actual_groups:
                    actual_groups[group] = 0
                actual_groups[group] += 1

    # Verify
    all_good = True
    for group, expected_count in expected_groups.items():
        actual_count = actual_groups.get(group, 0)

        if 'layers' in group:
            # For layers, we expect multiple parameters per layer
            # Each encoder layer has 7 params (4 attention + 3 mlp)
            # Each decoder layer has 11 params (4 self_attn + 4 cross_attn + 3 mlp)
            if 'encoder' in group:
                expected_total = 8 * 7  # 56
            else:
                expected_total = 8 * 11  # 88

            if actual_count == expected_total:
                status = "✓"
            else:
                status = "✗"
                all_good = False

            if verbose:
                print(f"  {status} {group:40s}: {actual_count:3d} params (expected ~{expected_total})")
        else:
            if actual_count >= expected_count:
                status = "✓"
            else:
                status = "✗"
                all_good = False

            if verbose:
                print(f"  {status} {group:40s}: {actual_count:3d} params")

    total_params = len(state_dict)
    if verbose:
        print()
        print(f"Total parameters in state_dict: {total_params}")

        # Calculate total size
        total_elements = sum(p.numel() for p in state_dict.values())
        print(f"Total elements: {total_elements:,}")

    if verbose:
        print("=" * 100)
        if all_good:
            print("✓ Verification PASSED")
        else:
            print("✗ Verification FAILED - some parameter groups are missing")
        print("=" * 100)

    return all_good


def main():
    """Main conversion script."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert JAX MT3 checkpoint to PyTorch')
    parser.add_argument(
        '--jax-checkpoint',
        default='musictranscription/checkpoints/mt3',
        help='Path to JAX checkpoint directory'
    )
    parser.add_argument(
        '--output',
        default='mt3_pytorch_checkpoint.pt',
        help='Output path for PyTorch checkpoint'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the conversion'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    verbose = not args.quiet

    try:
        # Convert
        state_dict = convert_checkpoint(
            args.jax_checkpoint,
            args.output,
            verbose=verbose,
        )

        # Verify if requested
        if args.verify:
            verify_conversion(state_dict, verbose=verbose)

        if verbose:
            print("\n✓ Conversion complete!")
            print(f"\nTo use the converted checkpoint:")
            print(f"  from mt3.pytorch_model import MT3Model, MT3Config")
            print(f"  import torch")
            print(f"  ")
            print(f"  model = MT3Model(MT3Config())")
            print(f"  state_dict = torch.load('{args.output}')")
            print(f"  model.load_state_dict(state_dict)")
            print(f"  model.eval()")

        return 0

    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
