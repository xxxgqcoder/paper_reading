import json
import os

from huggingface_hub import snapshot_download as hf_snapshot_download
from mineru.utils.enum_class import ModelPath
from modelscope import snapshot_download as ms_snapshot_download


def download_mineru_model(project_dir: str):
    """
    Download MinerU model weights and modify the config file.

    Args:
    - project_dir: The root directory of the MinerU project.
    """

    def download_mineru_model_weight(
        relative_path: str, repo_mode: str = "pipeline"
    ) -> str:
        """
        Download MinerU model weights from the specified repository.

        Args:
        - relative_path: The relative path of the model within the repository.
        - repo_mode: The repository mode, either 'pipeline' or 'vlm'.

        Returns:
        - The local cache directory where the model weights are stored.
        """
        model_source = os.getenv("MINERU_MODEL_SOURCE", "huggingface")
        print(f"downloading model from source: {model_source}, repo_mode: {repo_mode}")

        repo_mapping = {
            "pipeline": {
                "huggingface": ModelPath.pipeline_root_hf,
                "modelscope": ModelPath.pipeline_root_modelscope,
                "default": ModelPath.pipeline_root_hf,
            },
            "vlm": {
                "huggingface": ModelPath.vlm_root_hf,
                "modelscope": ModelPath.vlm_root_modelscope,
                "default": ModelPath.vlm_root_hf,
            },
        }

        if repo_mode not in repo_mapping:
            raise ValueError(
                f"Unsupported repo_mode: {repo_mode}, must be 'pipeline' or 'vlm'"
            )

        repo = repo_mapping[repo_mode].get(
            model_source, repo_mapping[repo_mode]["default"]
        )

        if model_source == "huggingface":
            snapshot_download = hf_snapshot_download
        elif model_source == "modelscope":
            snapshot_download = ms_snapshot_download
        else:
            raise ValueError(f"unknown repo type: {model_source}")

        cache_dir = None

        if repo_mode == "pipeline":
            relative_path = relative_path.strip("/")
            cache_dir = snapshot_download(
                repo, allow_patterns=[relative_path, relative_path + "/*"]
            )
        elif repo_mode == "vlm":
            if relative_path == "/":
                cache_dir = snapshot_download(repo)
            else:
                relative_path = relative_path.strip("/")
                cache_dir = snapshot_download(
                    repo, allow_patterns=[relative_path, relative_path + "/*"]
                )

        if not cache_dir:
            raise FileNotFoundError(
                f"Failed to download model: {relative_path} from {repo}"
            )
        return cache_dir

    # donwnload model
    model_paths = [
        ModelPath.doclayout_yolo,
        ModelPath.yolo_v8_mfd,
        ModelPath.unimernet_small,
        ModelPath.pytorch_paddle,
        ModelPath.layout_reader,
        ModelPath.slanet_plus,
        ModelPath.unet_structure,
        ModelPath.paddle_table_cls,
        ModelPath.paddle_orientation_classification,
    ]
    downloaded_model_dir = ""
    for model_path in model_paths:
        print(f"downloading model from: {model_path}")
        downloaded_model_dir = download_mineru_model_weight(
            model_path, repo_mode="pipeline"
        )
    print(f"donwloaded model path: {downloaded_model_dir}")

    # modify json config file
    config_file_path = os.path.join(project_dir, "magic-pdf.json")
    data = {
        "bucket_info": {
            "bucket-name-1": ["ak", "sk", "endpoint"],
            "bucket-name-2": ["ak", "sk", "endpoint"],
        },
        "latex-delimiter-config": {
            "display": {"left": "$$", "right": "$$"},
            "inline": {"left": "$", "right": "$"},
        },
        "llm-aided-config": {
            "title_aided": {
                "api_key": "your_api_key",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen2.5-32b-instruct",
                "enable": False,
            },
        },
        "mineru_tools_conf_json": config_file_path,
        "mineru_model_source": "local",
        "models-dir": {
            "pipeline": downloaded_model_dir,
        },
    }

    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"save modified config file to path: {config_file_path}")


if __name__ == "__main__":
    project_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    print(f"project directory: {project_dir}")

    download_mineru_model(project_dir)
    print("finish downloading MinerU model")
