# 🚀 PipeCraft Starter Kit

## GitHub Description

**PipeCraft**: WIP: A high-throughput data preprocessing pipeline framework for building scalable and efficient data workflows in a simple and modular way.

---

## 🛠️ README

### PipeCraft: High Throughput Data Preprocessing Pipelines

**PipeCraft** is a powerful, high-performance Python library designed for building efficient, scalable, and NUMA-aware data preprocessing pipelines. Tailored specifically for high-core, multi-NUMA-node servers, PipeCraft (tries to) ensures maximum CPU and memory utilization for intensive data workflows.

### ✨ Features

- **High Throughput**: Optimized for servers with hundreds of cores and large memory.
- **NUMA-Aware**: Memory and CPU affinity optimizations ensure minimal latency.
- **Modular and Extensible**: Easy-to-use API for defining custom data sources, processors, and sinks.
- **Efficient I/O**: Integrated optimizations for fast disk operations and minimal resource contention.
- **Multi-threaded and Multiprocessing Support**: Leverages parallelism effectively for CPU-bound and I/O-bound tasks.

### 🖥️ Installation

```bash
pip install pipecraft
```

### 🚦 Quickstart

Here's a simple pipeline example:

```python
from pipecraft.sources import DataSource
from pipecraft.processors import DataProcessor
from pipecraft.sinks import DataSink

class MySource(DataSource):
    def fetch(self, threadId):
        # Fetch your data here
        pass

class MyProcessor(DataProcessor):
    def process(self, data, threadId):
        # Process data here
        pass

class MySink(DataSink):
    def write(self, data, threadId):
        # Output data here
        pass

# Set up pipeline
source = MySource()
processor = MyProcessor(sources=[source])
sink = MySink(sources=[processor])

# Start pipeline
source.start()
processor.start()
sink.start()

# Wait for completion
sink.join()
```

### 🖥️ NUMA & CPU Affinity Example

Run your pipeline with NUMA bindings using `numactl`:

```bash
numactl --cpubind=0 --membind=0 python my_pipeline.py
```

### 📚 Documentation

Full documentation coming soon! Check out the examples folder for more use cases.

### 🔧 Contributing

Contributions are welcome! Please open issues and submit pull requests.

### 📄 License

MIT License
