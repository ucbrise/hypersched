import argparse


def make_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Shard.",
    )
    parser.add_argument(
        "--redis-address",
        default=None,
        help="The Redis address of the cluster.",
    )
    parser.add_argument(
        "--batch-size", default=1024, type=int, help="Batch per device."
    )
    parser.add_argument(
        "--num-iters", default=4, type=int, help="Number of iterations."
    )
    parser.add_argument(
        "--steps-per-iter", default=5, type=int, help="Steps per iteration."
    )
    # There is a difference between this and refresh_freq.
    parser.add_argument(
        "--trainable-id", default="optimus", type=str, help="Trainable class."
    )
    parser.add_argument("--upload", action="store_true", help="Upload or not.")
    parser.add_argument(
        "--prefetch", action="store_true", help="Prefetch data onto all nodes"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPUs")
    parser.add_argument("--tune", action="store_true", help="Use Tune.")
    parser.add_argument(
        "--ray-logs", default=None, help="Directory of Ray logs."
    )
    return parser
