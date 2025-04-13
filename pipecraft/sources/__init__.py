from abc import ABC, abstractmethod
from typing import Iterator, TypeVar, Generic, Type, get_args, get_origin, Iterable
import threading
import queue
import os
import traceback
from typing import TypeVar, Generic, Optional, List 
import time
from tqdm import tqdm

DataFrame = TypeVar("DataFrame")

NO_DATA = object()

class DataSource(Generic[DataFrame], Iterator[DataFrame], ABC):
    """
    Abstract class for a data source.
    Provides multi-threaded fetching with a queue-based buffer.
    Includes CPU affinity setting for worker threads.
    """
    def __init__(
        self,
        buffer_size: int = None,
        num_workers: int = None,
        verbose: bool = False,
        assigned_cores: Optional[List[int]] = None, # Added: Cores assigned to this source's process
        **kwargs,
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.assigned_cores = assigned_cores # Added

        self.__set_property_types__()

        self._num_workers = num_workers if num_workers is not None else (len(assigned_cores) if assigned_cores else (os.cpu_count() or 2))
        # Ensure num_workers doesn't exceed assigned cores if provided
        if self.assigned_cores and self._num_workers > len(self.assigned_cores):
             print(f"[DataSource Warning] num_workers ({self._num_workers}) > assigned_cores ({len(self.assigned_cores)}). Reducing num_workers.")
             self._num_workers = len(self.assigned_cores)

        self._buffer_size = buffer_size if buffer_size is not None else 2 * self._num_workers

        self.queue = queue.Queue(self._buffer_size)
        self._stop_event = threading.Event()
        self._threads = []
        self._closed = False
        self.lock = threading.Lock()
        
    def __set_property_types__(self) -> None:
        self.dataframe_class = None
        # Try to extract from __orig_class__:
        if hasattr(self, '__orig_class__'):
            args = get_args(self.__orig_class__)
            if args:
                self.dataframe_class = args[0]
        # Fallback to __orig_bases__:
        if self.dataframe_class is None:
            for base in getattr(self.__class__, '__orig_bases__', []):
                origin = get_origin(base)
                if origin is DataSource:
                    args = get_args(base)
                    if args:
                        self.dataframe_class = args[0]
                        break
        
    def start(self) -> None:
        # Launch worker threads
        for threadId in range(self._num_workers):
            thread = threading.Thread(target=self._worker, daemon=True, args=(threadId,))
            thread.start()
            self._threads.append(thread)
            
    def join(self) -> None:
        # Wait for all threads to finish
        for thread in self._threads:
            thread.join()
        self._threads.clear()
        self._closed = True

    def _worker(self, threadId) -> None:
        """
        Worker thread that repeatedly calls `fetch()` and places items into our queue.
        Sets CPU affinity for the thread if assigned_cores is provided.
        """
        if self.assigned_cores:
            try:
                # Use os.sched_setaffinity (0 means current thread)
                # This restricts the thread to run ONLY on the specified cores.
                os.sched_setaffinity(0, self.assigned_cores)
                if self.verbose and threadId == 0: # Log only once per source instance
                     print(f"[DataSource] Worker process {os.getpid()} (thread {threading.get_ident()}) pinned to cores: {self.assigned_cores}")
            except OSError as e:
                print(f"[DataSource Warning] Could not set CPU affinity for thread {threading.get_ident()} to {self.assigned_cores}: {e}")
            except AttributeError:
                 print(f"[DataSource Warning] os.sched_setaffinity not available on this platform. Skipping thread pinning.")

        while not self._stop_event.is_set():
            try:
                data = self.fetch(threadId=threadId)
                if data == NO_DATA:
                    # Signal that this worker is done
                    break
                # Allow fetch to return None without breaking the worker loop,
                # DataProcessor uses None to signal exhaustion, but the worker
                # should only stop on NO_DATA or error.
                if data is not None: # Check explicitly for not None
                    self.queue.put(data)
            except StopIteration: # Added: Handle StopIteration from upstream in DataProcessor
                 if self.verbose:
                     print(f"[DataSource] Worker {threadId} encountered StopIteration from upstream.")
                 break # Stop this worker if upstream is exhausted
            except Exception as e:
                print(f"[DataSource] Worker error in thread {threadId}: {e}")
                traceback.print_exc()
                self._stop_event.set() # Signal others to stop on error
                break

        if self.verbose:
            print(f"[DataSource] Worker thread {threadId} stopping.")
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the maximum number of items this source can provide.
        """
        pass
            
    @abstractmethod
    def fetch(self, threadId) -> DataFrame:
        """
        Called by each worker thread to get one new item of data.
        Return NO_DATA to signal no more data to fetch for this source instance.
        Returning None might indicate a temporary lack of data or upstream exhaustion (handled by DataProcessor).
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Stop the data source and free resources.
        """
        pass

    def __next__(self) -> DataFrame:
        """
        Get the next item from our queue in a blocking manner.
        Raises StopIteration if no more items and workers are stopped.
        """
        while True:
            if self._stop_event.is_set() and self.queue.empty():
                raise StopIteration
            try:
                return self.queue.get(timeout=1)
            except queue.Empty:
                # The queue is empty but maybe not truly done
                # Keep looping until the stop event is definitively set and queue is empty
                continue
    
    def has_data(self) -> bool:
        """
        Check if there's data available in the queue.
        Returns True if there's at least one item, False otherwise.
        """
        return not self.queue.empty()

    def __iter__(self) -> Iterator[DataFrame]:
        return self

    def __close__(self) -> None:
        """Helper that sets stop-event, waits for threads to exit, and calls `close()`."""
        if not self._closed:
            self._stop_event.set()
            # Join threads with timeout
            join_timeout = 5.0
            start_time = time.time()
            for t in self._threads:
                 remaining_time = join_timeout - (time.time() - start_time)
                 if remaining_time <= 0:
                      print(f"[DataSource Warning] Timeout expired joining threads. Thread {t.ident} might still be running.")
                      break
                 t.join(timeout=remaining_time)

            # Clear queue more safely
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
                except Exception as e:
                     print(f"[DataSource Warning] Error clearing queue: {e}")
                     break # Avoid potential infinite loop on unexpected errors

            try:
                self.close() # Call the subclass's close method
            except Exception as e:
                 print(f"[DataSource Error] Exception during self.close(): {e}")
                 traceback.print_exc()

            self._threads.clear() # Ensure threads list is cleared
            self._closed = True

    def __enter__(self):
        # Make sure start() is called when entering context if not already started
        if not self._threads:
             self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__close__()

    @property
    def dataframe_type(self) -> Type[DataFrame]:
        """Return the type of DataFrame this DataSource provides."""
        return self.dataframe_class