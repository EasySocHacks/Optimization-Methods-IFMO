import numpy as np

from HW2.error_calculator import Error, SquaredErrorCalculator
from HW2.optimization import DefaultOptimization, Optimization


def format_bytes(bytes):
    bytes
    if abs(bytes) < 1000:
        return str(bytes) + "B"
    elif abs(bytes) < 1e6:
        return str(round(bytes / 1e3, 2)) + "kB"
    elif abs(bytes) < 1e9:
        return str(round(bytes / 1e6, 2)) + "MB"
    else:
        return str(round(bytes / 1e9, 2)) + "GB"


class BenchmarkStorage:
    def __init__(self):
        self.benchmark_results = []
        self.configs = []

    def add_benchmark_config(self, batch_size: str = 'svd', normalized: bool = False,
                             optimiser: Optimization = DefaultOptimization(),
                             error: Error = SquaredErrorCalculator()
                             ):
        self.configs.append((batch_size, normalized, optimiser, error))
        self.benchmark_results.append({
            "time": [],
            "mem": [],
            "smape": []
        })
        return len(self.configs) - 1

    def add_benchmark_result(self, index, benchmark_result):
        self.benchmark_results[index]['time'].append(benchmark_result['time'])
        self.benchmark_results[index]['mem'].append(benchmark_result['maximum-after'])
        self.benchmark_results[index]['smape'].append(benchmark_result['smape'])

    def get_benchmark_results(self, index):
        config_name = self.configs[index]
        return 'Benchmark results for config <{}>:\n\tMean time:{}\n\tMean memory:{}\n\tMean SMAPE value:{}\n'.format(
            config_name,
            np.mean(self.benchmark_results[index]['time']),
            format_bytes(np.mean(self.benchmark_results[index]['mem'])),
            np.mean(self.benchmark_results[index]['smape']))
