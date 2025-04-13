from abc import ABC, abstractmethod
from typing import Iterator
from pipecraft.utils import logger
from typing import Iterator, TypeVar, Generic, Type, get_args, get_origin, Iterable
import threading
import queue
from pipecraft.sources import DataSource
from pipecraft.processors import DataProcessor
from abc import ABC, abstractmethod
from typing import Iterable, TypeVar, Generic
from tqdm import tqdm

InputDataFrame = TypeVar("DataInputDataFrameFrame")

# DataSink is generic over DataFrame, just like DataProcessor
class DataSink(DataProcessor[InputDataFrame, None], ABC):
    """
    A DataSink is the final consumer in a data processing pipeline.
    It inherits the multi-threaded, queued processing from DataProcessor,
    but instead of a transformation that returns a new DataFrame, it calls
    a user-defined sink function to consume each data item.
    
    Optionally, the sink() function may return a confirmation or status which
    will be forwarded through the queue.
    """
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        source_len = self.__len__()
        self._progress_bar = None
        if source_len is not None:
            # Initialize tqdm progress bar
            print(f"[DataSource: {self.__class__.__name__}] Detected source length: {source_len}")
            self._progress_bar = tqdm(total=source_len, desc=self.__class__.__name__, unit="item")
    
    @abstractmethod
    def write(self, data: InputDataFrame, threadId) -> None:
        """
        User-defined function to consume a data item.
        For example, writing the data to a file, database, or any other endpoint.
        """
        pass

    def process(self, data: InputDataFrame, threadId) -> None:
        """
        Instead of performing a transformation and returning a new item,
        this method calls the sink() function with the data.
        You may optionally return the original data (or a status) if you want
        to pass something downstream; otherwise, the sink side effect is enough.
        """
        self.write(data, threadId=threadId)
        with self.lock:
            if self._progress_bar is not None:
                self._progress_bar.update(1)
        return None