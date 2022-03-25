import numpy as np

from HW2.error_calculator import Error, SquaredErrorCalculator
from HW2.optimization import DefaultOptimization, Optimization


def format_bytes(bytes):
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

    def add_benchmark_config(self, batch_size: str = 'sgd', scale: float = 1,
                             optimiser: Optimization = DefaultOptimization(),
                             error: Error = SquaredErrorCalculator()
                             ):
        self.configs.append((batch_size, scale, optimiser, error))
        self.benchmark_results.append({
            "time": [],
            "mem": [],
            "smape": [],
            "gradient_calls": []
        })
        return len(self.configs) - 1

    def add_benchmark_result(self, index, benchmark_result):
        self.benchmark_results[index]['time'].append(benchmark_result['time'])
        self.benchmark_results[index]['mem'].append(benchmark_result['maximum-after'])
        self.benchmark_results[index]['smape'].append(benchmark_result['smape'])
        self.benchmark_results[index]['gradient_calls'].append(benchmark_result['gradient_call_count'])
        self.benchmark_results[index]['iterations'].append(benchmark_result['iterations'])

    def get_benchmark_results(self, index):
        config_name = self.configs[index]
        return 'Benchmark results for config <{}>:\n\tMean time:{}\n\tMean memory:{}\n\tMean SMAPE value:{}\nMean gradient calls:{}'.format(
            config_name,
            np.mean(self.benchmark_results[index]['time']),
            format_bytes(np.mean(self.benchmark_results[index]['mem'])),
            np.mean(self.benchmark_results[index]['smape']),
            np.mean(self.benchmark_results[index]['gradient_calls']),
            np.mean(self.benchmark_results[index]['iterations']))

    def get_benchmark_results_arrayed(self, index):
        return np.array([self.configs[index], np.mean(self.benchmark_results[index]['time']),
                         format_bytes(np.mean(self.benchmark_results[index]['mem'])),
                         np.mean(self.benchmark_results[index]['smape']),
                         np.mean(self.benchmark_results[index]['gradient_calls']),
                         np.mean(self.benchmark_results[index]['iterations'])], dtype=object)
