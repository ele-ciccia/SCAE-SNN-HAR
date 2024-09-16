import argparse

parser = argparse.ArgumentParser(description='Description of your script')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset file')
parser.add_argument('--hyperparam_to_optimize', nargs='+', type=float, required=True, help='List of values for the hyperparameter to optimize')


args = parser.parse_args()


dataset_path = args.dataset_path
hyperparam_to_optimize = args.hyperparam_to_optimize

# Your code to handle the dataset and hyperparameter optimization goes here

if __name__ == "__main__":
    main()
