import numpy as np


def data_regression_2D_1():
  """
    uniform, noncentered, y = x1 * x2
  """
  n = 2
  N = 4000

  X1 = np.random.random(N) - 0.5
  X2 = np.random.random(N) - 0.5

  y = X1 * X2 + np.random.random(N) / 5
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_5():
  """
    uniform, noncentered, y = x1 + x2
  """

  n = 2
  N = 4000

  X1 = np.random.random(N) - 0.5
  X2 = np.random.random(N) - 0.5

  y = X1 + X2 + np.random.random(N) / 5
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_2():
  """
  gauss, centered, y = x1 * x2
  """
  n = 2
  N = 4000

  X1 = np.random.randn(N) * 10
  X2 = np.random.randn(N) * 10

  y = X1 * X2 + np.random.randn(N)
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_6():
  """
  gauss, centered, y = x1 + x2
  """
  n = 2
  N = 4000

  X1 = np.random.randn(N) * 10
  X2 = np.random.randn(N) * 10

  y = X1 + X2 + np.random.randn(N)
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_3():
  """
  uniform, centered, y = x1 * x2
  """
  n = 2
  N = 4000

  X1 = np.random.random(N)
  X2 = np.random.random(N)

  y = X1 * X2 + np.random.random(N) / 5
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_7():
  """
  uniform, centered, y = x1 + x2
  """
  n = 2
  N = 4000

  X1 = np.random.random(N)
  X2 = np.random.random(N)

  y = X1 + X2 + np.random.random(N) / 5
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_4():
  """
  uniform, noncentered, y = x1 * x2
  """
  n = 2
  N = 4000

  X1 = np.random.randn(N) * 10 + 5
  X2 = np.random.randn(N) * 10 -10

  y = X1 * X2 + np.random.randn(N)
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_regression_2D_8():
  """
  uniform, noncentered, y = x1 + x2
  """
  n = 2
  N = 4000

  X1 = np.random.randn(N) * 10 + 5
  X2 = np.random.randn(N) * 10 -10

  y = X1 + X2 + np.random.randn(N)
  X = np.vstack([X1, X2]).transpose(1, 0)

  return X, y

def data_classification_2D_3():
    """
    classification, interected
    """
    n = 2
    l = 2000

  # Параметры
    width = 50         # Ширина прямоугольника
    height = 50        # Высота прямоугольника
    cols = 6         # Количество столбцов сетки
    rows = 6          # Количество строк сетки
    points_per_cell = 50  # Количество точек в каждой ячейке

    # Вычисление размеров ячейки
    cell_width = width / cols
    cell_height = height / rows

    # Генерация данных
    X = []
    y = []

    for i in range(cols):
        for j in range(rows):
            # Генерация координат точек внутри ячейки (i, j)
            x_points = np.random.uniform(i * cell_width, (i + 1) * cell_width, points_per_cell)
            y_points = np.random.uniform(j * cell_height, (j + 1) * cell_height, points_per_cell)
            
            # Определение класса (шахматный порядок)
            class_label = (i + j) % 2
            
            # Сохранение данных
            X.extend(np.column_stack((x_points, y_points)))
            y.extend([class_label] * points_per_cell)

    X = np.array(X)
    y = np.array(y)

    return X, y

def data_classification_2D_4():
    """
    classification, non-interacted
    """
    n = 2
    l = 2000

    # Параметры
    width = 50        # Ширина прямоугольника
    height = 50         # Высота прямоугольника
    cols = 6           # Количество столбцов
    rows = 6           # Количество строк
    points_per_cell = 50  # Количество точек в каждой ячейке

    # Размеры ячейки
    cell_width = width / cols
    cell_height = height / rows

    # Генерация данных
    X = []
    y = []

    for i in range(cols):
        for j in range(rows):
            # Генерация случайных точек внутри ячейки (i, j)
            x_points = np.random.uniform(i * cell_width, (i + 1.2) * cell_width, points_per_cell)
            y_points = np.random.uniform(j * cell_height, (j + 1) * cell_height, points_per_cell)
            
            # Класс определяется только номером столбца (i)
            class_label = i % 2  # Четные столбцы — 0, нечетные — 1
            
            X.extend(np.column_stack((x_points, y_points)))
            y.extend([class_label] * points_per_cell)

    X = np.array(X)
    y = np.array(y)

    return X, y