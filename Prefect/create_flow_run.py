from train_classifiers import _parse_args, train_flow


def main():
    cmd_args = _parse_args()
    params = vars(cmd_args)
    train_flow(params)


if __name__ == "__main__":
    main()
