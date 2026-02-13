#!/usr/bin/env python3
"""
Expert Cache Analyzer for MoE Models

This script analyzes expert usage statistics and estimates the performance
improvement from caching frequently-used experts in GPU memory.

Usage:
    python analyze-expert-data.py --json expert_stats.json [options]

Options:
    --json FILE              Path to expert stats JSON file (required)
    --gpu-vram GB            Available GPU VRAM in GB (default: 16)
    --expert-size MB         Size per expert in MB (default: auto-detect from model)
    --accel-index FLOAT      GPU acceleration index (speedup factor)
    --model-name NAME        Model name for reference
    --total-experts N        Total number of experts per layer
    --output FILE            Output file for detailed report

Example:
    # First, run baseline tests to get acceleration index:
    # Full GPU: llama-bench -ngl 99 ...
    # Full CPU: llama-bench -ngl 0 ...
    # acceleration_index = gpu_speed / cpu_speed
    
    python analyze-expert-data.py \\
        --json expert_stats.json \\
        --gpu-vram 16 \\
        --accel-index 12.5 \\
        --model-name "gpt-oss-120b" \\
        --total-experts 128
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class ExpertStats:
    """Statistics for a single expert in a layer."""
    expert_id: int
    activations: int
    percentage: float


@dataclass
class LayerStats:
    """Statistics for a single layer."""
    layer_id: int
    total_tokens: int
    experts: List[ExpertStats] = field(default_factory=list)
    
    def get_sorted_experts(self) -> List[ExpertStats]:
        """Return experts sorted by activation count (descending)."""
        return sorted(self.experts, key=lambda e: e.activations, reverse=True)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "unknown"
    total_experts: int = 32  # Experts per layer
    experts_per_token: int = 4  # Top-k experts activated per token
    num_layers: int = 24
    hidden_dim: int = 2880
    intermediate_dim: int = 0  # FFN intermediate dimension
    quantization: str = "Q4_K_M"
    
    def get_expert_size_mb(self) -> float:
        """
        Estimate expert size in MB.
        
        Each expert has 3 weight matrices (gate, up, down) of shape [hidden_dim, intermediate_dim].
        For Q4_K_M quantization, each weight is ~0.5 bytes on average.
        """
        if self.intermediate_dim == 0:
            # Estimate intermediate dim as 4x hidden dim (common for MoE)
            self.intermediate_dim = self.hidden_dim * 4
        
        # 3 matrices per expert: gate, up, down
        # Each matrix: hidden_dim x intermediate_dim
        params_per_expert = 3 * self.hidden_dim * self.intermediate_dim
        
        # Quantization factor (bytes per parameter)
        quant_factors = {
            "Q4_K_M": 0.5,
            "Q5_K_M": 0.6,
            "Q6_K": 0.7,
            "Q8_0": 1.0,
            "FP16": 2.0,
        }
        bytes_per_param = quant_factors.get(self.quantization, 0.5)
        
        size_bytes = params_per_expert * bytes_per_param
        return size_bytes / (1024 * 1024)  # Convert to MB


@dataclass
class CacheAnalysis:
    """Results of cache analysis."""
    experts_in_cache: int
    cache_hit_rate: float
    estimated_speedup: float
    vram_used_gb: float
    hot_experts: Dict[int, List[int]]  # layer_id -> list of expert IDs


# Predefined model configurations
MODEL_CONFIGS = {
    "gpt-oss-20b": ModelConfig(
        name="gpt-oss-20b",
        total_experts=32,
        experts_per_token=4,
        num_layers=24,
        hidden_dim=2880,
        quantization="Q4_K_M",
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        total_experts=128,
        experts_per_token=4,
        num_layers=36,
        hidden_dim=2880,
        quantization="Q8_0",
    ),
    "qwen3-next-80b": ModelConfig(
        name="qwen3-next-80b",
        total_experts=512,
        experts_per_token=10,
        num_layers=48,
        hidden_dim=2048,
        quantization="Q4_K_M",
    ),
}


def load_expert_stats(json_path: str) -> Dict[int, LayerStats]:
    """Load expert statistics from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    layers = {}
    for layer_data in data.get("layers", []):
        layer_id = layer_data["layer_id"]
        total_tokens = layer_data["total_tokens"]
        
        experts = []
        for exp_data in layer_data.get("experts", []):
            experts.append(ExpertStats(
                expert_id=exp_data["expert_id"],
                activations=exp_data["activations"],
                percentage=exp_data["percentage"],
            ))
        
        layers[layer_id] = LayerStats(
            layer_id=layer_id,
            total_tokens=total_tokens,
            experts=experts,
        )
    
    return layers


def calculate_acceleration_index(gpu_pp_speed: float, gpu_tg_speed: float,
                                  cpu_pp_speed: float, cpu_tg_speed: float) -> float:
    """
    Calculate GPU acceleration index from benchmark results.
    
    The acceleration index represents how much faster GPU is compared to CPU.
    We use a weighted average of prompt processing and token generation speeds.
    
    Args:
        gpu_pp_speed: GPU prompt processing speed (tokens/s)
        gpu_tg_speed: GPU token generation speed (tokens/s)
        cpu_pp_speed: CPU prompt processing speed (tokens/s)
        cpu_tg_speed: CPU token generation speed (tokens/s)
    
    Returns:
        Acceleration index (e.g., 12.5 means GPU is 12.5x faster)
    """
    # Weight token generation more heavily as it's the typical bottleneck
    pp_weight = 0.3
    tg_weight = 0.7
    
    pp_accel = gpu_pp_speed / cpu_pp_speed if cpu_pp_speed > 0 else 1.0
    tg_accel = gpu_tg_speed / cpu_tg_speed if cpu_tg_speed > 0 else 1.0
    
    return pp_weight * pp_accel + tg_weight * tg_accel


def analyze_cache(layers: Dict[int, LayerStats], 
                  model: ModelConfig,
                  gpu_vram_gb: float,
                  accel_index: float,
                  expert_size_mb: Optional[float] = None) -> CacheAnalysis:
    """
    Analyze optimal cache configuration.
    
    Args:
        layers: Expert usage statistics per layer
        model: Model configuration
        gpu_vram_gb: Available GPU VRAM in GB
        accel_index: GPU acceleration index
        expert_size_mb: Override expert size in MB
    
    Returns:
        CacheAnalysis with recommendations
    """
    if expert_size_mb is None:
        expert_size_mb = model.get_expert_size_mb()
    
    # Calculate how many experts we can cache
    vram_bytes = gpu_vram_gb * 1024 * 1024 * 1024
    expert_bytes = expert_size_mb * 1024 * 1024
    
    # Reserve some VRAM for non-expert weights and activations
    usable_vram = vram_bytes * 0.7  # 70% for expert cache
    
    max_experts_cache = int(usable_vram / expert_bytes)
    
    # Distribute cache budget across layers
    # Each layer gets an equal share of the cache
    experts_per_layer = max(1, max_experts_cache // model.num_layers)
    
    # Select hot experts for each layer based on usage frequency
    hot_experts = {}
    total_activations = 0
    cached_activations = 0
    
    for layer_id, layer_stats in layers.items():
        sorted_experts = layer_stats.get_sorted_experts()
        
        # Select top-k experts for this layer
        layer_hot = []
        layer_total = sum(e.activations for e in sorted_experts)
        layer_cached = 0
        
        for i, expert in enumerate(sorted_experts):
            if i < experts_per_layer:
                layer_hot.append(expert.expert_id)
                layer_cached += expert.activations
        
        hot_experts[layer_id] = layer_hot
        total_activations += layer_total
        cached_activations += layer_cached
    
    # Calculate cache hit rate
    cache_hit_rate = cached_activations / total_activations if total_activations > 0 else 0.0
    
    # Estimate speedup
    # If expert is in cache: GPU speed (accel_index)
    # If expert is not in cache: CPU speed (1.0)
    # Weighted average based on cache hit rate
    estimated_speedup = cache_hit_rate * accel_index + (1 - cache_hit_rate) * 1.0
    
    # Calculate actual VRAM used
    actual_experts = sum(len(experts) for experts in hot_experts.values())
    vram_used_gb = (actual_experts * expert_bytes) / (1024 * 1024 * 1024)
    
    return CacheAnalysis(
        experts_in_cache=actual_experts,
        cache_hit_rate=cache_hit_rate,
        estimated_speedup=estimated_speedup,
        vram_used_gb=vram_used_gb,
        hot_experts=hot_experts,
    )


def print_report(model: ModelConfig, 
                 analysis: CacheAnalysis,
                 layers: Dict[int, LayerStats],
                 accel_index: float,
                 expert_size_mb: float):
    """Print detailed analysis report."""
    print("=" * 70)
    print("EXPERT CACHE ANALYSIS REPORT")
    print("=" * 70)
    print()
    
    print("MODEL CONFIGURATION")
    print("-" * 40)
    print(f"  Model:              {model.name}")
    print(f"  Total experts:       {model.total_experts} per layer")
    print(f"  Experts per token:   {model.experts_per_token}")
    print(f"  Number of layers:    {model.num_layers}")
    print(f"  Expert size:         {expert_size_mb:.2f} MB")
    print()
    
    print("PERFORMANCE BASELINE")
    print("-" * 40)
    print(f"  GPU Acceleration:    {accel_index:.2f}x")
    print()
    
    print("CACHE ANALYSIS")
    print("-" * 40)
    print(f"  Experts in cache:    {analysis.experts_in_cache}")
    print(f"  Cache hit rate:      {analysis.cache_hit_rate * 100:.1f}%")
    print(f"  VRAM used:           {analysis.vram_used_gb:.2f} GB")
    print()
    
    print("ESTIMATED PERFORMANCE")
    print("-" * 40)
    print(f"  Estimated speedup:   {analysis.estimated_speedup:.2f}x vs CPU-only")
    print(f"  Effective speed:     {analysis.estimated_speedup / accel_index * 100:.1f}% of full GPU")
    print()
    
    # Print top experts per layer
    print("HOT EXPERTS PER LAYER (Top 5)")
    print("-" * 40)
    for layer_id in sorted(analysis.hot_experts.keys())[:5]:  # Show first 5 layers
        hot = analysis.hot_experts[layer_id]
        layer_stats = layers[layer_id]
        sorted_experts = layer_stats.get_sorted_experts()
        
        print(f"  Layer {layer_id}: ", end="")
        for i, expert_id in enumerate(hot[:5]):
            # Find the expert's activation count
            for e in sorted_experts:
                if e.expert_id == expert_id:
                    print(f"E{expert_id}({e.percentage:.1f}%) ", end="")
                    break
        print()
    
    if len(analysis.hot_experts) > 5:
        print(f"  ... and {len(analysis.hot_experts) - 5} more layers")
    print()
    
    # Print speedup comparison table
    print("SPEEDUP COMPARISON")
    print("-" * 40)
    print("  Configuration          | Speedup | vs Full GPU")
    print("  -----------------------|---------|------------")
    print(f"  Full CPU (ngl=0)       |   1.00x |      0.0%")
    print(f"  Expert Cache (est.)    | {analysis.estimated_speedup:6.2f}x | {analysis.estimated_speedup / accel_index * 100:6.1f}%")
    print(f"  Full GPU (ngl=99)      | {accel_index:6.2f}x |    100.0%")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert usage and estimate cache performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--json", required=True, help="Path to expert stats JSON file")
    parser.add_argument("--gpu-vram", type=float, default=16.0, help="Available GPU VRAM in GB")
    parser.add_argument("--expert-size", type=float, default=None, help="Expert size in MB (auto-calculated if not set)")
    parser.add_argument("--accel-index", type=float, default=10.0, help="GPU acceleration index")
    parser.add_argument("--model-name", default="gpt-oss-20b", help="Model name for configuration")
    parser.add_argument("--total-experts", type=int, default=None, help="Override total experts per layer")
    parser.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--output", default=None, help="Output file for detailed report")
    
    # Baseline benchmark results (for calculating acceleration index)
    parser.add_argument("--gpu-pp", type=float, default=None, help="GPU prompt processing speed (t/s)")
    parser.add_argument("--gpu-tg", type=float, default=None, help="GPU token generation speed (t/s)")
    parser.add_argument("--cpu-pp", type=float, default=None, help="CPU prompt processing speed (t/s)")
    parser.add_argument("--cpu-tg", type=float, default=None, help="CPU token generation speed (t/s)")
    
    args = parser.parse_args()
    
    # Load expert statistics
    try:
        layers = load_expert_stats(args.json)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {args.json}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get model configuration
    model = MODEL_CONFIGS.get(args.model_name, ModelConfig(name=args.model_name))
    
    # Override with command-line arguments
    if args.total_experts:
        model.total_experts = args.total_experts
    if args.num_layers:
        model.num_layers = args.num_layers
    
    # Calculate acceleration index from benchmark results if provided
    accel_index = args.accel_index
    if all([args.gpu_pp, args.gpu_tg, args.cpu_pp, args.cpu_tg]):
        accel_index = calculate_acceleration_index(
            args.gpu_pp, args.gpu_tg, args.cpu_pp, args.cpu_tg
        )
        print(f"Calculated acceleration index: {accel_index:.2f}x")
    
    # Get expert size
    expert_size_mb = args.expert_size if args.expert_size else model.get_expert_size_mb()
    
    # Run analysis
    analysis = analyze_cache(
        layers=layers,
        model=model,
        gpu_vram_gb=args.gpu_vram,
        accel_index=accel_index,
        expert_size_mb=expert_size_mb,
    )
    
    # Print report
    print_report(
        model=model,
        analysis=analysis,
        layers=layers,
        accel_index=accel_index,
        expert_size_mb=expert_size_mb,
    )
    
    # Save detailed report if requested
    if args.output:
        output_data = {
            "model": {
                "name": model.name,
                "total_experts": model.total_experts,
                "experts_per_token": model.experts_per_token,
                "num_layers": model.num_layers,
                "expert_size_mb": expert_size_mb,
            },
            "cache": {
                "experts_in_cache": analysis.experts_in_cache,
                "cache_hit_rate": analysis.cache_hit_rate,
                "vram_used_gb": analysis.vram_used_gb,
                "estimated_speedup": analysis.estimated_speedup,
            },
            "hot_experts": analysis.hot_experts,
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Detailed report saved to: {args.output}")


if __name__ == "__main__":
    main()
