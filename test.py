import os
from robocasa.scripts.download_datasets import download_datasets
from robocasa.scripts.dataset_scripts.playback_dataset import playback_dataset
from robocasa.utils.dataset_registry_utils import get_ds_path

TASK = "PickPlaceCounterToCabinet"

dataset = get_ds_path(TASK, source="human", split="pretrain")

if not os.path.exists(dataset):
    print("Downloading dataset...")
    download_datasets(tasks=[TASK], split=["pretrain"], source=["human"])

playback_dataset(
    dataset=dataset,
    use_actions=False,
    use_abs_actions=False,
    use_obs=False,
    filter_key=None,
    n=1,
    render=True,
    render_image_names=["robot0_agentview_center"],
    camera_height=512,
    camera_width=768,
    video_path=False,
    video_skip=5,
    extend_states=True,
    first=False,
    verbose=True,
)
