from pydantic import BaseModel

class Config(BaseModel):
    board_size: int = 9
    max_stones: int = 32

    wandb: bool = True

    seed: int = 42
    max_num_iters: int = 400

    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256

    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    gamma: float = 0.99  # discount factor for the continuing/infinite-horizon return

    # eval params
    eval_interval: int = 5

    save_interval: int = 5
