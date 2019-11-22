import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Keras MNIST Example")
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", help="Use GPU in training."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of jobs to run concurrently (default: 1)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="threads used in operations (default: 2)",
    )
    parser.add_argument(
        "--steps",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--kernel1",
        type=int,
        default=3,
        help="Size of first kernel (default: 3)",
    )
    parser.add_argument(
        "--kernel2",
        type=int,
        default=3,
        help="Size of second kernel (default: 3)",
    )
    parser.add_argument(
        "--poolsize", type=int, default=2, help="Size of Pooling (default: 2)",
    )
    parser.add_argument(
        "--dropout1",
        type=float,
        default=0.25,
        help="Size of first kernel (default: 0.25)",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Size of Hidden Layer (default: 128)",
    )
    parser.add_argument(
        "--dropout2",
        type=float,
        default=0.5,
        help="Size of first kernel (default: 0.5)",
    )
    return parser
