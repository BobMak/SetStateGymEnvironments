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
    entry_point="gym_lunar_lander_set_state.envs:MyLunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)