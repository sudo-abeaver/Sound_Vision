#!/usr/bin/env python3
"""
Benchmark TFLite models on Raspberry Pi
Tests inference speed and memory usage
"""

import time
import numpy as np
import psutil
import sys
import os

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("ERROR: Neither tflite_runtime nor tensorflow.lite found!")
        sys.exit(1)

def benchmark_model(model_path, name="Model", runs=50):
    """Benchmark a TFLite model"""
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"Path: {model_path}")
    print(f"{'='*60}")
    
    # Load model
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return None
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Input shape: {input_shape}")
    print(f"Input dtype: {input_dtype}")
    print(f"Number of inputs: {len(input_details)}")
    print(f"Number of outputs: {len(output_details)}")
    
    # Create random input
    if input_dtype == np.float32:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
    elif input_dtype == np.uint8:
        dummy_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
    else:
        dummy_input = np.zeros(input_shape, dtype=input_dtype)
    
    # Warmup runs
    print("\nWarming up...")
    for _ in range(5):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    
    # Benchmark
    print(f"Running {runs} inference iterations...")
    
    times = []
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    for i in range(runs):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end = time.time()
        times.append(end - start)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{runs}")
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    # Statistics
    times = np.array(times) * 1000  # Convert to ms
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}")
    print(f"Mean inference time:   {np.mean(times):.2f} ms")
    print(f"Median inference time: {np.median(times):.2f} ms")
    print(f"Std deviation:         {np.std(times):.2f} ms")
    print(f"Min time:              {np.min(times):.2f} ms")
    print(f"Max time:              {np.max(times):.2f} ms")
    print(f"FPS (mean):            {1000/np.mean(times):.2f}")
    print(f"Memory usage:          {mem_after:.1f} MB (Δ {mem_after-mem_before:.1f} MB)")
    
    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'fps': 1000/np.mean(times),
        'memory_mb': mem_after
    }

def main():
    print("="*60)
    print("TFLite Model Benchmark Tool")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    
    # System info
    print(f"\nSystem Information:")
    print(f"  CPU count: {psutil.cpu_count()}")
    print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB")
    
    # Models to benchmark
    models = [
        ("../models/scene_int8.tflite", "Scene Classifier"),
        ("../models/detect_int8.tflite", "Object Detector"),
    ]
    
    results = {}
    
    for model_path, name in models:
        result = benchmark_model(model_path, name, runs=50)
        if result:
            results[name] = result
    
    # Summary
    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, result in results.items():
            print(f"{name:20s}: {result['mean_ms']:6.2f} ms ({result['fps']:5.1f} FPS)")
        
        # Combined inference time
        total_time = sum(r['mean_ms'] for r in results.values())
        combined_fps = 1000 / total_time if total_time > 0 else 0
        print(f"\nCombined inference:  {total_time:6.2f} ms ({combined_fps:5.1f} FPS)")
        print(f"\nNote: This is theoretical max FPS assuming perfect parallelization.")
        print(f"      Actual performance will be lower due to frame capture overhead.")

if __name__ == "__main__":
    main()

