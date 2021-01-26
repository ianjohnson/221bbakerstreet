from gym.envs.registration import register

register(
    id='BakerStreet-v1',
    entry_point='gym_221bbakerstreet.environments:BakerStreetEnvironment',
)
