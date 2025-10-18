# TensorProv

**TensorProv** is a Python framework designed to **capture and query the provenance of data preparation pipelines**.  
It enables users to trace how data transformations, such as filtering, merging, and projection, affect the resulting datasets — providing detailed provenance information that supports reproducibility, debugging, and performance analysis.

This repository is organized into three main directories:

- **`src/`** — Contains the source code of TensoProv, developed in Python.  
- **`tests/`** — Includes unit and integration tests used to verify that the system operates correctly.  
- **`examples/`** — Provides a collection of Jupyter notebooks demonstrating the use of TensoProv across multiple scenarios.  
  These examples cover three key use cases: **COMPAS**, **German Credit**, and **Census** datasets.  
  In addition, the directory includes experiments for assessing the performance of provenance capture during join operations using the **TPC-DI Benchmark**.

---

## Local Package Installation

To install TensoProv as a local package, execute the following commands from the root of the repository:

```bash
rmdir /s /q dist  
rmdir /s /q build  
rmdir /s /q src\tensor_prov.egg-info

python -m build  
pip install -e .
