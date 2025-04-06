import subprocess
import re
import os
import psutil # Keep psutil for fallback/verification if needed
from typing import List, Dict, Optional
    
def get_local_partition(objs, world_size, rank):
    """
    Get the local partition of the objects based on the world size and rank.
    """
    partition_size = len(objs) // world_size
    start = rank * partition_size
    end = (rank + 1) * partition_size if rank != world_size - 1 else len(objs)
    return objs[start:end]


def optimize_io():
    # Set larger read-ahead for disk
    try:
        with open('/proc/sys/vm/page-cluster', 'w') as f:
            f.write('16')  # Larger page cluster size
            
        # Adjust I/O scheduler if needed
        devices = os.listdir('/sys/block')
        for dev in devices:
            if dev.startswith('sd') or dev.startswith('nvme'):
                scheduler_path = f'/sys/block/{dev}/queue/scheduler'
                if os.path.exists(scheduler_path):
                    with open(scheduler_path, 'w') as f:
                        f.write('deadline')  # Use deadline scheduler for better throughput
    except Exception as e:
        print(f"Failed to optimize I/O: {e}")

def parse_numactl_hardware() -> Dict[int, List[int]]:
    """
    Parses the output of `numactl --hardware` to get a mapping
    of NUMA node IDs to lists of CPU core IDs belonging to that node.

    Returns:
        A dictionary where keys are node IDs (int) and values are
        lists of core IDs (int) associated with that node.
        Returns an empty dictionary if numactl fails or output is unexpected.
    """
    node_cores: Dict[int, List[int]] = {}
    try:
        # Execute numactl --hardware
        result = subprocess.run(
            ['numactl', '--hardware'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout

        # Regex to find lines like "node X cpus: Y Z A B ..."
        # It handles single-digit, multi-digit core IDs, and ranges (though ranges are less common now)
        # We will primarily rely on space-separated lists.
        cpu_line_regex = re.compile(r"^\s*node\s+(\d+)\s+cpus:\s*(.*)", re.MULTILINE)

        for match in cpu_line_regex.finditer(output):
            node_id = int(match.group(1))
            cpu_list_str = match.group(2).strip()

            cores = []
            # Split the cpu list string by space and convert to int
            # This handles space-separated lists like "0 1 2 3..."
            try:
                 # This might fail if numactl output uses ranges like '0-7'
                 # Add more robust parsing if needed for ranges.
                 cores = [int(cpu) for cpu in cpu_list_str.split()]
            except ValueError:
                 print(f"[Warning] Could not parse core list for node {node_id}: '{cpu_list_str}'. Skipping node.")
                 continue # Skip this node if parsing fails

            if cores:
                node_cores[node_id] = sorted(cores) # Store sorted list

        if not node_cores:
             print("[Warning] `numactl --hardware` parsing yielded no node-core mapping.")

    except FileNotFoundError:
        print("[Error] `numactl` command not found. Please install numactl (e.g., `sudo apt install numactl` or `sudo yum install numactl`). Cannot perform NUMA-aware assignment.")
        return {}
    except subprocess.CalledProcessError as e:
        print(f"[Error] `numactl --hardware` failed with error code {e.returncode}: {e.stderr}")
        return {}
    except Exception as e:
        print(f"[Error] Failed to parse `numactl --hardware` output: {e}")
        return {}

    return node_cores

def get_cpu_assignments_numa(num_processes: int) -> List[List[int]]:
    """
    Assigns CPU cores to processes in a NUMA-aware manner.

    It attempts to distribute processes evenly across NUMA nodes and assigns
    cores belonging to the same node to the processes running on that node.

    Args:
        num_processes: The total number of worker processes to assign cores for.

    Returns:
        A list of lists, where each inner list contains the CPU core IDs
        assigned to a process. The length of the outer list is `num_processes`.
        Returns empty lists if NUMA info cannot be obtained or assignments fail.
    """
    node_to_cores = parse_numactl_hardware()

    if not node_to_cores:
        print("[Warning] Falling back to non-NUMA core assignment (sequential across all cores).")
        # Fallback: Use psutil to get all cores sequentially (similar to old logic)
        try:
             total_cores = psutil.cpu_count(logical=True) # Use logical in fallback for safety
             if not total_cores: return [[]] * num_processes
             all_core_ids = list(range(total_cores))
             cores_per_process = max(1, total_cores // num_processes)
             assignments = []
             core_idx = 0
             for i in range(num_processes):
                 assigned = all_core_ids[core_idx : core_idx + cores_per_process]
                 # Handle last process potentially getting fewer if not divisible
                 if i == num_processes -1:
                      assigned = all_core_ids[core_idx:]
                 if not assigned and i < num_processes: # Assign at least one if possible and needed
                      if core_idx < total_cores: assigned = [all_core_ids[core_idx]]
                      else: assigned = [] # No more cores left

                 assignments.append(assigned)
                 core_idx += len(assigned) # Move index correctly
             return assignments

        except Exception as e:
             print(f"Error during fallback core assignment: {e}")
             return [[]] * num_processes


    # --- NUMA-aware assignment logic ---
    num_nodes = len(node_to_cores)
    node_ids = sorted(node_to_cores.keys())
    assignments = [[] for _ in range(num_processes)]
    process_idx = 0

    # Determine how many processes go to each node
    base_procs_per_node = num_processes // num_nodes
    extra_procs = num_processes % num_nodes

    print(f"Distributing {num_processes} processes across {num_nodes} NUMA nodes.")
    print(f"Node core map: {node_to_cores}")

    for i, node_id in enumerate(node_ids):
        procs_on_this_node = base_procs_per_node + (1 if i < extra_procs else 0)
        if procs_on_this_node == 0:
            continue

        cores_on_node = node_to_cores[node_id]
        if not cores_on_node:
            print(f"[Warning] No cores found for Node {node_id}. Skipping assignment for this node.")
            continue

        # Divide cores on this node among the processes assigned to it
        base_cores_per_proc_node = len(cores_on_node) // procs_on_this_node
        extra_cores_node = len(cores_on_node) % procs_on_this_node

        if base_cores_per_proc_node == 0 and extra_cores_node < procs_on_this_node:
             print(f"[Warning] Not enough cores ({len(cores_on_node)}) on Node {node_id} "
                   f"to assign at least one core to each of the {procs_on_this_node} processes planned for it. "
                   "Processes might receive empty core lists for this node.")
             # Handle this - maybe assign cores round-robin? For now, some might get empty.

        core_start_idx = 0
        print(f"  Assigning {procs_on_this_node} processes to Node {node_id} (cores: {len(cores_on_node)})")

        for j in range(procs_on_this_node):
            cores_for_this_proc = base_cores_per_proc_node + (1 if j < extra_cores_node else 0)
            core_end_idx = core_start_idx + cores_for_this_proc

            # Ensure indices are valid and slice cores
            actual_end_idx = min(core_end_idx, len(cores_on_node))
            assigned_node_cores = cores_on_node[core_start_idx:actual_end_idx]

            if process_idx < num_processes:
                if not assigned_node_cores and len(cores_on_node) > 0:
                     # If calculation resulted in zero cores, but node has cores, maybe assign at least one?
                     # This can happen if procs_on_this_node > len(cores_on_node)
                     # Simple fix: assign the next available core if possible (could lead to unevenness)
                     if core_start_idx < len(cores_on_node):
                          assigned_node_cores = [cores_on_node[core_start_idx]]
                          actual_end_idx = core_start_idx + 1 # Increment end index for next iteration
                     # print(f"[Debug] Assigning single core {assigned_node_cores} to process {process_idx} on node {node_id} due to low core count.")


                assignments[process_idx] = assigned_node_cores
                print(f"    Process {process_idx} assigned to Node {node_id} with cores: {assigned_node_cores}")
                process_idx += 1
            else:
                 # Should not happen if logic is correct, but good failsafe
                 print("[Warning] Generated more core assignments than requested processes.")
                 break

            core_start_idx = actual_end_idx # Use actual_end_idx for next start

    # Sanity check: Ensure all processes got *some* assignment, even if empty
    if process_idx < num_processes:
         print(f"[Warning] Could only generate assignments for {process_idx} out of {num_processes} requested processes.")
         # Remaining processes in `assignments` will have empty lists `[]`

    print(f"Final CPU Assignments (NUMA aware): {assignments}")
    return assignments
