import os
import time
from pathlib import Path
import concurrent.futures
from functools import partial

# Your original function
def original_gather_video_paths_fast(input_dir, output_dir):
    video_paths = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if file.endswith(".mp4"):
                rel_path = os.path.relpath(root, input_dir)
                video_input = os.path.join(root, file)
                video_output_dir = os.path.join(output_dir, rel_path)
                video_output = os.path.join(video_output_dir, file)
                if not os.path.isfile(video_output):
                    video_paths.append((video_input, video_output))
    # Sort the paths to ensure consistent order
    video_paths.sort()
    return video_paths

# Implementation from previous response
def pathlib_gather_video_paths_fast(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    video_paths = []
    
    # Use glob pattern matching for faster file discovery
    for video_input in sorted(input_path.glob('**/*.mp4')):
        # Calculate relative path once
        rel_path = video_input.relative_to(input_path)
        video_output = output_path / rel_path
        
        # Fast existence check
        if not video_output.exists():
            video_paths.append((str(video_input), str(video_output)))
    
    return video_paths

# Define process_path function outside the main function to make it picklable
def process_path(input_path, input_dir, output_dir):
    input_prefix_len = len(input_dir) + 1  # +1 for the trailing slash
    rel_path = input_path[input_prefix_len:]
    output_path = os.path.join(output_dir, rel_path)
    
    # Fast existence check
    try:
        os.stat(output_path)
        exists = True
    except FileNotFoundError:
        exists = False
        
    if not exists:
        return (input_path, output_path)
    return None

def find_mp4_files_chunk(dir_chunk, suffix=".mp4"):
    """Process a chunk of directories to find mp4 files."""
    results = []
    for dir_path in dir_chunk:
        try:
            # Get all items in directory without recursion
            with os.scandir(dir_path) as it:
                for entry in it:
                    if entry.is_file() and entry.name.endswith(suffix):
                        results.append(entry.path)
                    elif entry.is_dir():
                        # Add directory to be processed
                        results.extend(find_mp4_files_chunk([entry.path], suffix))
        except (PermissionError, FileNotFoundError):
            pass
    return results

def chunk_directories(root_dir, chunk_size=100):
    """Split directories into chunks for parallel processing."""
    dirs = [root_dir]
    chunk_dirs = []
    
    # Collect top-level directories first
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                dirs.append(entry.path)
    
    # Create chunks
    for i in range(0, len(dirs), chunk_size):
        chunk_dirs.append(dirs[i:i + chunk_size])
    
    return chunk_dirs if chunk_dirs else [dirs]

def gather_video_paths_ultra_fast(input_dir, output_dir, workers=32, chunk_size=100):
    """Ultra-optimized function to gather video paths."""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Use all available CPUs by default
    workers = workers or os.cpu_count()
    
    # Step 1: Split the directory tree into chunks for parallel processing
    dir_chunks = chunk_directories(input_dir, chunk_size)
    
    # Step 2: Find all mp4 files in parallel
    video_input_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        chunk_results = executor.map(find_mp4_files_chunk, dir_chunks)
        for result in chunk_results:
            video_input_paths.extend(result)
    
    # Step 3: Sort input paths (single sort operation)
    video_input_paths.sort()
    
    # Step 4: Prepare output paths and filter existing files
    video_paths = []
    
    # Process paths in batches to avoid GIL contention
    batch_size = 1000
    for i in range(0, len(video_input_paths), batch_size):
        batch = video_input_paths[i:i+batch_size]
        
        # Use partial to bind the fixed parameters
        process_func = partial(process_path, input_dir=input_dir, output_dir=output_dir)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            # Process batch in parallel with the bound function
            results = executor.map(process_func, batch)
            for result in results:
                if result:
                    video_paths.append(result)
    
    return video_paths

# Global cache to optimize repeated file lookups
class FileExistenceCache:
    def __init__(self, capacity=100000):
        self.cache = {}
        self.capacity = capacity
    
    def file_exists(self, path):
        if path in self.cache:
            return self.cache[path]
        
        # Fast existence check
        try:
            os.stat(path)
            exists = True
        except FileNotFoundError:
            exists = False
        
        # Manage cache size
        if len(self.cache) >= self.capacity:
            # Simple eviction - clear half the cache when full
            keys = list(self.cache.keys())[:self.capacity//2]
            for k in keys:
                del self.cache[k]
        
        self.cache[path] = exists
        return exists

# Create a global cache instance
file_cache = FileExistenceCache()

def gather_video_paths_cached(input_dir, output_dir, workers=None):
    """Version with file existence caching for repeated calls"""
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    # Find all mp4 files using a more efficient approach
    video_input_paths = []
    for root, _, files in os.walk(input_dir, followlinks=False):
        for file in files:
            if file.endswith(".mp4"):
                video_input_paths.append(os.path.join(root, file))
    
    # Sort once
    video_input_paths.sort()
    
    # Process in larger batches with caching
    video_paths = []
    input_prefix_len = len(input_dir) + 1
    
    for input_path in video_input_paths:
        rel_path = input_path[input_prefix_len:]
        output_path = os.path.join(output_dir, rel_path)
        
        if not file_cache.file_exists(output_path):
            video_paths.append((input_path, output_path))
    
    return video_paths

def process_file(video_input, input_path, output_path):
    rel_path = video_input.relative_to(input_path)
    video_output = output_path / rel_path
    if not video_output.exists():
        return (str(video_input), str(video_output))
    return None

def gather_video_paths_fast(input_dir, output_dir, max_workers=64):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all mp4 files upfront
    all_videos = sorted(input_path.glob('**/*.mp4'))
    
    # Process files in parallel
    processor = partial(process_file, input_path=input_path, output_path=output_path)
    video_paths = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(processor, all_videos)
        for result in results:
            if result:
                video_paths.append(result)
    
    return video_paths

def canonicalize_paths(paths):
    """Normalize paths for comparison"""
    return [(os.path.normpath(i), os.path.normpath(o)) for i, o in paths]

def verify_results(input_dir, output_dir):
    """Verify that all implementations return equivalent results"""
    print(f"Testing with input_dir={input_dir}, output_dir={output_dir}")
    
    # Time and collect results from original function
    start_time = time.time()
    original_results = original_gather_video_paths_fast(input_dir, output_dir)
    original_time = time.time() - start_time
    original_count = len(original_results)
    print(f"Original function found {original_count} files in {original_time:.2f} seconds")
    
    # Time and collect results from original function
    start_time = time.time()
    threaded_results = gather_video_paths_fast(input_dir, output_dir)
    threaded_time = time.time() - start_time
    threaded_count = len(threaded_results)
    print(f"Gather fast (multi-threaded) function found {threaded_count} files in {threaded_time:.2f} seconds")
    
    # Time and collect results from pathlib version
    start_time = time.time()
    pathlib_results = pathlib_gather_video_paths_fast(input_dir, output_dir)
    pathlib_time = time.time() - start_time
    pathlib_count = len(pathlib_results)
    print(f"Pathlib version found {pathlib_count} files in {pathlib_time:.2f} seconds")
    
    # Time and collect results from ultra-fast version
    start_time = time.time()
    ultra_results = gather_video_paths_ultra_fast(input_dir, output_dir)
    ultra_time = time.time() - start_time
    ultra_count = len(ultra_results)
    print(f"Ultra-fast version found {ultra_count} files in {ultra_time:.2f} seconds")
    
    # Time and collect results from cached version
    start_time = time.time()
    cached_results = gather_video_paths_cached(input_dir, output_dir)
    cached_time = time.time() - start_time
    cached_count = len(cached_results)
    print(f"Cached version found {cached_count} files in {cached_time:.2f} seconds")
    
    # Normalize paths for comparison
    original_normalized = set(canonicalize_paths(original_results))
    threaded_normalized = set(canonicalize_paths(threaded_results))
    pathlib_normalized = set(canonicalize_paths(pathlib_results))
    ultra_normalized = set(canonicalize_paths(ultra_results))
    cached_normalized = set(canonicalize_paths(cached_results))
    
    # Check if results are identical
    pathlib_matches = original_normalized == pathlib_normalized
    ultra_matches = original_normalized == ultra_normalized
    cached_matches = original_normalized == cached_normalized
    threaded_matches = original_normalized == threaded_normalized
    
    print("\nResults Comparison:")
    print(f"Pathlib version matches original: {pathlib_matches}")
    print(f"Ultra-fast version matches original: {ultra_matches}")
    print(f"Cached version matches original: {cached_matches}")
    print(f"Threaded version matches original: {threaded_matches}")
    
    # If any don't match, report differences
    if not (pathlib_matches and ultra_matches and cached_matches):
        print("\nAnalyzing differences...")
        
        if not pathlib_matches:
            original_only = original_normalized - pathlib_normalized
            pathlib_only = pathlib_normalized - original_normalized
            print(f"Pathlib missing {len(original_only)} files found in original")
            print(f"Pathlib found {len(pathlib_only)} files not in original")
            if len(original_only) > 0:
                print(f"Example missing: {next(iter(original_only))}")
            if len(pathlib_only) > 0:
                print(f"Example extra: {next(iter(pathlib_only))}")
        
        if not ultra_matches:
            original_only = original_normalized - ultra_normalized
            ultra_only = ultra_normalized - original_normalized
            print(f"Ultra-fast missing {len(original_only)} files found in original")
            print(f"Ultra-fast found {len(ultra_only)} files not in original")
            if len(original_only) > 0:
                print(f"Example missing: {next(iter(original_only))}")
            if len(ultra_only) > 0:
                print(f"Example extra: {next(iter(ultra_only))}")
        
        if not cached_matches:
            original_only = original_normalized - cached_normalized
            cached_only = cached_normalized - original_normalized
            print(f"Cached missing {len(original_only)} files found in original")
            print(f"Cached found {len(cached_only)} files not in original")
            if len(original_only) > 0:
                print(f"Example missing: {next(iter(original_only))}")
            if len(cached_only) > 0:
                print(f"Example extra: {next(iter(cached_only))}")
    
    # Calculate speed improvements
    print("\nPerformance Improvements:")
    print(f"Pathlib version: {original_time/pathlib_time:.2f}x faster")
    print(f"Ultra-fast version: {original_time/ultra_time:.2f}x faster")
    print(f"Cached version: {original_time/cached_time:.2f}x faster")
    print(f"Threaded version: {original_time/threaded_time:.2f}x faster")
    
    # Second run of cached version to show caching benefit
    print("\nRunning cached version a second time to show caching benefit:")
    start_time = time.time()
    cached_results_2 = gather_video_paths_cached(input_dir, output_dir)
    cached_time_2 = time.time() - start_time
    print(f"Cached version (2nd run): {len(cached_results_2)} files in {cached_time_2:.2f} seconds")
    print(f"Cache improvement: {cached_time/cached_time_2:.2f}x faster than first run")
    
    return {
        "all_match": pathlib_matches and ultra_matches and cached_matches,
        "speedups": {
            "pathlib": original_time/pathlib_time,
            "ultra": original_time/ultra_time,
            "cached": original_time/cached_time,
            "cached_2nd_run": cached_time/cached_time_2
        }
    }

# Example usage
if __name__ == "__main__":
    # Replace with your actual directories
    input_dir='/home/mrwhite0racle/persist/data/vox2/train_output/'
    output_dir='/home/mrwhite0racle/persist/data/vox2/train_2/'

    
    results = verify_results(input_dir, output_dir)
    
    if results["all_match"]:
        print("\n✅ SUCCESS: All implementations return identical results!")
        print(f"Best speedup: {max(results['speedups'].values()):.2f}x faster")
    else:
        print("\n❌ WARNING: Results differ between implementations!")