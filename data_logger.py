from gradio import utils
import os
import csv
import huggingface_hub


def log_data(hf_token: str, dataset_name: str, private=True):
    path_to_dataset_repo = huggingface_hub.create_repo(
        repo_id=dataset_name,
        token=hf_token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )
    flagging_dir = "flagged"
    dataset_dir = os.path.join(flagging_dir, dataset_name)
    repo = huggingface_hub.Repository(
        local_dir=dataset_dir,
        clone_from=path_to_dataset_repo,
        use_auth_token=hf_token,
    )
    repo.git_pull(lfs=True)
    log_file = os.path.join(dataset_dir, "data.csv")

    def log_function(data):
        repo.git_pull(lfs=True)

        with open(log_file, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            for row in data:
                writer.writerow(utils.sanitize_list_for_csv(row))

        with open(log_file, "r", encoding="utf-8") as csvfile:
            line_count = len([None for row in csv.reader(csvfile)]) - 1

        repo.push_to_hub(commit_message="Flagged sample #{}".format(line_count))

        return line_count

    return log_function
