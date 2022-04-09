from gym.envs.registration import register

register(
    id='ContinuousCache-v0',
    entry_point='gym_cachingSystem_env.envs:ContinuousCache',
)

register(
    id='foo-Test-v0',
    entry_point='gym_cachingSystem_env.envs:TestEnv',
)

register(
    id='foo-TestMarl-v0',
    entry_point='gym_cachingSystem_env.envs:TestMarlEnv',
)