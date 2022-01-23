from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()

register(
    id="LunarLanderSetState-v0",
    entry_point="gym_set_state.envs:MyLunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="LunarLanderContinuousSetState-v0",
    entry_point="gym.envs.box2d:LunarLander",
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id="BipedalWalkerSetState-v0",
    entry_point="gym_set_state.envs:MyBipedalWalker",
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id="BipedalWalkerHardcoreSetState-v0",
    entry_point="gym.envs.box2d:BipedalWalker",
    kwargs={"hardcore": True},
    max_episode_steps=2000,
    reward_threshold=300,
)