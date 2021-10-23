from gym.envs.registration import register

register(
    id='ContinuousCache-v0',
    entry_point='gym_cachingSystem_env.envs:ContinuousCache',
)

register(
    id='foo-SelectSame-v0',
    entry_point='gym_cachingSystem_env.envs:SelectSameEnv',
)

register(
    id='foo-Action1-v0',
    entry_point='gym_cachingSystem_env.envs:Action1Env',
)

register(
    id='foo-Test-v0',
    entry_point='gym_cachingSystem_env.envs:TestEnv',
)

register(
    id='foo-TestMarl-v0',
    entry_point='gym_cachingSystem_env.envs:TestMarlEnv',
)