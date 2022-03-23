# class Error(ABC):
#     @abstractmethod
#     def calc_error(self, f, point):
#         pass
#
#     @abstractmethod
#     def generate(self, point):
#         pass
#
#     @abstractmethod
#     def gradient(self, f):
#         pass
#
#
# class AbsErrorCalculator(Error):
#     def calc_error(self, f, point):
#         return np.abs(f(point[0]) - point[1])
#
#     def generate(self, point_y):
#         return lambda a, b: lambda x: np.abs(a * x + b - point_y)
#
#     def gradient(self, point_y):
#         return lambda a, b: lambda x: a * np.sign(a * x + b - point_y)
#
#
# class SquaredErrorCalculator(Error):
#     def calc_error(self, f, point):
#         return np.square(f(point[0]) - point[1])
#
#     def generate(self, point_y):
#         return lambda a, b: lambda x: (a * x + b - point_y) ** 2
#
#     def gradient(self, point_y):
#         return lambda a, b: lambda x: 2 * a * (a * x + b - point_y) * 2
