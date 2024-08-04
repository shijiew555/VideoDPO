import os
import argparse
import shutil


def clean_dir(dirname):
    """Check directories and delete or print based on specific conditions."""
    for root, dirnames, filenames in os.walk(dirname):
        if "checkpoints" in dirnames:
            checkpoints_path = os.path.join(root, "checkpoints")
            trainstep_checkpoints_path = os.path.join(
                checkpoints_path, "trainstep_checkpoints"
            )

            if (
                len(os.listdir(checkpoints_path)) == 1
                and os.path.isdir(trainstep_checkpoints_path)
                and len(os.listdir(trainstep_checkpoints_path)) == 0
            ):
                # Remove the root directory
                print(root)
                shutil.rmtree(root)
            else:
                # Print the root directory path
                print(
                    root,
                    os.listdir(checkpoints_path),
                    os.listdir(trainstep_checkpoints_path),
                )
        # else:
        #     print(f"root {root} not contain checkpoints")


def main():
    parser = argparse.ArgumentParser(description="List contents of a directory.")
    parser.add_argument('-d',"--dirname", type=str, help="The name of the directory")

    args = parser.parse_args()

    clean_dir(args.dirname)


if __name__ == "__main__":
    main()
