import numpy as np

np.random.seed(42)


def test_numpy_random_seed():
    random_cube = np.random.randint(low=0, high=101, size=[5, 5, 5], dtype=np.int64)
    test_cube = np.array(
        [
            [
                [51, 92, 14, 71, 60],
                [20, 82, 86, 74, 74],
                [87, 99, 23, 2, 21],
                [52, 1, 87, 29, 37],
                [1, 63, 59, 20, 32],
            ],
            [
                [75, 57, 21, 88, 48],
                [90, 58, 41, 91, 59],
                [79, 14, 61, 61, 46],
                [61, 50, 54, 63, 2],
                [100, 50, 6, 20, 72],
            ],
            [
                [38, 17, 3, 88, 59],
                [13, 8, 89, 52, 1],
                [83, 91, 59, 70, 43],
                [7, 46, 34, 77, 80],
                [35, 49, 3, 1, 5],
            ],
            [
                [53, 3, 53, 92, 62],
                [17, 89, 43, 33, 73],
                [61, 99, 13, 94, 47],
                [14, 71, 77, 86, 61],
                [39, 84, 79, 81, 52],
            ],
            [
                [23, 25, 88, 59, 40],
                [28, 14, 44, 64, 88],
                [70, 8, 87, 0, 7],
                [87, 62, 10, 80, 7],
                [34, 34, 32, 4, 40],
            ],
        ],
        dtype=np.int64,
    )  # result random cube with seed 42, prob. 100^125 = 1e250 = we are done here

    assert random_cube.shape == (5, 5, 5)
    assert random_cube.min() >= 0
    assert random_cube.max() <= 100
    assert random_cube.dtype == np.int64

    assert test_cube.shape == (5, 5, 5)
    assert test_cube.min() >= 0
    assert test_cube.max() <= 100
    assert test_cube.dtype == np.int64

    assert np.array_equal(random_cube, test_cube)
