from gym.envs.registration import register
register(
    id='mini-warehouse-v0',
    entry_point='warehouse.envs:MiniWarehouse',
)