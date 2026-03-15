"""Neural network implementations."""

from ..utils.config import NetworkConfig
from .conv_net import ConvNetWrapper
from .othello_net import OthelloNetWrapper
from .simple_net import SimpleNetWrapper


def create_model(game, config: NetworkConfig, lr: float = 0.001, weight_decay: float = 0.0,
                  max_grad_norm: float = 0.0, value_loss_weight: float = 1.0):
    """Factory: create the right model wrapper based on config.network_type.

    Args:
        game: Game instance (must implement get_board_size, get_action_size, get_board_shape).
        config: NetworkConfig with network_type ('mlp', 'cnn', or 'othellonet').
        lr: Learning rate for the optimizer.
        weight_decay: L2 regularization weight.
        max_grad_norm: Maximum gradient norm for clipping. 0.0 = disabled.
        value_loss_weight: Weight for value loss in total loss.

    Returns:
        SimpleNetWrapper, ConvNetWrapper, or OthelloNetWrapper.
    """
    board_size = game.get_board_size()
    action_size = game.get_action_size()

    if config.network_type == "mlp":
        model = SimpleNetWrapper(board_size, action_size, config, lr=lr)
    elif config.network_type == "cnn":
        board_shape = game.get_board_shape()
        model = ConvNetWrapper(board_size, action_size, config, lr=lr, board_shape=board_shape)
        if weight_decay > 0:
            model.weight_decay = weight_decay
            model.optimizer = __import__('torch').optim.Adam(model.net.parameters(), lr=lr, weight_decay=weight_decay)
    elif config.network_type == "othellonet":
        board_shape = game.get_board_shape()
        model = OthelloNetWrapper(board_size, action_size, config, lr=lr, board_shape=board_shape)
    else:
        raise ValueError(f"Unknown network_type '{config.network_type}'. Use 'mlp', 'cnn', or 'othellonet'.")

    # Set training parameters on the model wrapper
    model._max_grad_norm = max_grad_norm
    model._value_loss_weight = value_loss_weight
    return model
