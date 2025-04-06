import psutil
from concurrent.futures import ProcessPoolExecutor
import os
import gc
import traceback
from typing import List, Optional
from utils import get_cpu_assignments_numa, get_local_partition


# Modified function to accept core assignments and pin the process
def run_pipeline_proc(
    pipeline_opts: dict,
    pipeline: callable,
    
    assigned_cores: List[int], # Changed: Expect list of cores
    num_workers_per_process: int, # Renamed for clarity
    world_size: int,
    rank: int, # Keep rank for logging/debugging
    use_wandb: bool = False,
    verbose: bool = False,
):
    """Runs one instance of the pipeline, pinned to specific cores."""
    actual_num_workers = num_workers_per_process # Start with requested

    # --- Pin this entire process to the assigned cores ---
    if assigned_cores:
        try:
            p = psutil.Process()
            p.cpu_affinity(assigned_cores)
            current_affinity = p.cpu_affinity() # Get actual affinity
            print(f"Rank {rank} (PID {os.getpid()}) pinned to cores: {current_affinity} (Requested: {assigned_cores})")

            # Adjust num_workers based on the *actual* number of cores successfully assigned
            if not current_affinity:
                 print(f"Rank {rank}: Warning - Process affinity is empty after setting! Cannot use workers.")
                 actual_num_workers = 0 # Or handle error appropriately
            elif actual_num_workers > len(current_affinity):
                 print(f"Rank {rank}: Adjusting num_workers from {actual_num_workers} to {len(current_affinity)} based on actual core affinity.")
                 actual_num_workers = len(current_affinity)
            elif actual_num_workers <= 0:
                 print(f"Rank {rank}: Requested {actual_num_workers} workers, adjusting to 1.")
                 actual_num_workers = 1 # Ensure at least 1 worker if cores assigned

        except AttributeError: # Handle platforms where cpu_affinity might not exist
             print(f"Rank {rank} (PID {os.getpid()}): CPU affinity setting not available on this platform. Proceeding without process pinning.")
             # Keep requested num_workers if pinning fails/unavailable
        except Exception as e:
            print(f"Rank {rank} (PID {os.getpid()}): Error setting/checking CPU affinity: {e}")
            # Decide whether to proceed or halt; proceeding without pinning
            assigned_cores = [] # Clear assigned cores if pinning failed critically
            # Keep requested num_workers

    else:
        print(f"Rank {rank} (PID {os.getpid()}): No cores assigned, not pinning process.")
        # Ensure num_workers is at least 1 if not specified or zero
        if actual_num_workers <= 0: actual_num_workers = 1


    # Check if we ended up with zero workers - this shouldn't happen if logic is sound
    if actual_num_workers <= 0:
         print(f"Rank {rank}: ERROR - Number of workers is {actual_num_workers}. Cannot proceed.")
         return # Exit this process


    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                 project="Voxceleb-prep",
                 name=f"pipeline_rank{rank}_pid{os.getpid()}",
                 config={
                     "rank": rank,
                     "assigned_cores": assigned_cores,
                     "num_workers_per_process": num_workers_per_process,
                     **pipeline_opts # Log other options
                 },
                 reinit=True, # Allow multiple wandb runs in one script execution
                 # mode="disabled" # Uncomment to disable wandb logging easily
            )
            print(f"Rank {rank}: WandB initialized.")
        except ImportError:
            print("Rank {rank}: 'wandb' library not found. Skipping WandB initialization.")
            use_wandb = False # Disable if import failed
        except Exception as e:
             print(f"Rank {rank}: Failed to initialize WandB: {e}")
             use_wandb = False # Disable on error


    # --- Retrieve options ---
    # --- Create the pipeline stages, passing assigned_cores ---
    # Use context managers for cleaner setup/teardown
    try:
        # Run the pipeline
        pipeline(
            rank=rank,
            world_size=world_size,
            num_workers_per_process=actual_num_workers, # Use actual number of workers
            wandb_run=wandb_run,
            assigned_cores=assigned_cores, # Pass the assigned cores
            verbose=verbose,
            **pipeline_opts # Unpack other options
        )
    except Exception as e:
        print(f"FATAL ERROR in Rank {rank} (PID {os.getpid()}) pipeline execution: {e}")
        traceback.print_exc()
    finally:
        # --- Clean up WandB ---
        if wandb_run:
            wandb_run.finish()
            print(f"Rank {rank}: WandB finished.")
        print(f"Rank {rank} (PID {os.getpid()}) finished execution.")
        # Explicit GC call at the end of the process might help release memory faster
        gc.collect()


# Modified main execution function
def run_pipeline(
    pipeline_opts: dict,
    pipeline: callable,
    
    verbose: bool = False, # Added verbose flag
    
    num_processes: int = 1,
    num_workers_per_process: int = 1,
    cores_per_process: Optional[int] = None, # Added: Explicit control over cores per process
    use_wandb: bool = False,
):
    """
    Sets up and runs the processing pipeline using a ProcessPoolExecutor,
    assigning specific CPU cores to each process.
    """
    print("Starting pipeline...")
    print(f"Config: num_processes={num_processes}, num_workers_per_process={num_workers_per_process}, cores_per_process={cores_per_process}")


    # --- Determine Core Assignments using NUMA ---
    all_assignments = get_cpu_assignments_numa(num_processes)

    if not any(all_assignments): # Check if all assignments are empty
         print("CRITICAL WARNING: CPU assignments are empty for all processes. Check numactl or system configuration. Proceeding without pinning, performance will be impacted.")
         # Fallback: allow processes to run unpinned with a default number of workers
         if num_workers_per_process <= 0: num_workers_per_process = max(1, (os.cpu_count() or 2) // num_processes) # Basic fallback worker count

    # Create the process pool
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []
        for rank in range(num_processes):
            assigned_cores_for_rank = all_assignments[rank] if rank < len(all_assignments) else []

            # Determine num_workers for this specific rank
            # If workers requested explicitly (>0), use that as max, otherwise default to num cores assigned.
            workers_for_rank = num_workers_per_process
            num_cores_assigned = len(assigned_cores_for_rank)

            if workers_for_rank <= 0:
                # Default: Use all assigned cores for workers (if any cores assigned)
                workers_for_rank = max(1, num_cores_assigned) # Ensure at least 1 if cores assigned
            elif num_cores_assigned > 0 and workers_for_rank > num_cores_assigned:
                 print(f"Rank {rank}: Requested {workers_for_rank} workers, but only {num_cores_assigned} cores assigned. Limiting workers to {num_cores_assigned}.")
                 workers_for_rank = num_cores_assigned
            elif num_cores_assigned == 0 and workers_for_rank > 0:
                 print(f"Rank {rank}: Warning - No cores assigned, but {workers_for_rank} workers requested. Running unpinned with {workers_for_rank} workers.")
                 # Keep requested workers, but assigned_cores_for_rank remains []
            elif workers_for_rank <= 0: # Handles case where assigned_cores=0 and requested workers=0
                 workers_for_rank = 1 # Default to 1 worker if no cores and no request


            # local_paths = get_local_partition(video_paths, num_processes, rank)
            for key in pipeline_opts:
                if type(pipeline_opts[key]) == list:
                    # Split the list into chunks for each process
                    local_objs = get_local_partition(pipeline_opts[key], num_processes, rank)
                    pipeline_opts[key] = local_objs
                    print(f"Rank {rank}: {key} has {len(local_objs)} items out of {len(pipeline_opts[key])} total.")
            print(f"Submitting Rank {rank}: Workers={workers_for_rank}, Cores={assigned_cores_for_rank}")
            futures.append(executor.submit(
                run_pipeline_proc,
                pipeline_opts=pipeline_opts,
                pipeline=pipeline,
                rank=rank,
                world_size=num_processes,
                verbose=verbose,
                assigned_cores=assigned_cores_for_rank,
                num_workers_per_process=workers_for_rank,
                use_wandb=use_wandb if rank == 0 else False, # Only first rank uses wandb
            ))

        # Wait for all processes to finish and handle potential exceptions
        print("Waiting for processes to complete...")
        results = []
        for rank, future in enumerate(futures):
            try:
                result = future.result() # Wait for completion and get result (None in this case)
                results.append(result)
                print(f"Rank {rank} completed successfully.")
            except Exception as e:
                print(f"!!! Rank {rank} execution failed with exception: {e}")
                traceback.print_exc()
                # Decide if you want to stop other processes on failure
                # executor.shutdown(wait=False, cancel_futures=True) # Uncomment to cancel others on error
        print(f"All {len(futures)} processes finished.")
