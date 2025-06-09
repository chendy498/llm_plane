import numpy as np
from .env_base import BaseEnv
from ..tasks import SingleCombatTask, SingleCombatDodgeMissileTask, HierarchicalSingleCombatDodgeMissileTask, \
    HierarchicalSingleCombatShootTask, SingleCombatShootMissileTask, HierarchicalSingleCombatTask
from ..human_task.HumanSingleCombatTask import  HumanSingleCombatTask


class SingleCombatEnv(BaseEnv):
    """
    SingleCombatEnv is an one-to-one competitive environment.
    """
    def __init__(self, config_name: str):
        super().__init__(config_name)
        # Env-Specific initialization here!
        assert len(self.agents.keys()) == 2, f"{self.__class__.__name__} only supports 1v1 scenarios!"
        self.init_states = None

    def load_task(self):
        taskname = getattr(self.config, 'task', None)
        if taskname == 'singlecombat':#改成控制
            self.task = SingleCombatTask(self.config)
        elif taskname == 'hierarchical_singlecombat':
            self.task = HierarchicalSingleCombatTask(self.config)
        elif taskname == 'singlecombat_dodge_missile':
            self.task = SingleCombatDodgeMissileTask(self.config)
        elif taskname == 'singlecombat_shoot':#改成进攻
            self.task = SingleCombatShootMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_dodge_missile':
            self.task = HierarchicalSingleCombatDodgeMissileTask(self.config)
        elif taskname == 'hierarchical_singlecombat_shoot':#改成总的
            self.task = HierarchicalSingleCombatShootTask(self.config)
        elif taskname == 'HumanSingleCombat':
            self.task = HumanSingleCombatTask(self.config)
        else:
            raise NotImplementedError(f"Unknown taskname: {taskname}")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.reset_simulators()
        self.task.reset(self)
        obs = self.get_obs()
        return self._pack(obs)

    # def reset_simulators(self):
    #     # switch side
    #     if self.init_states is None:
    #         self.init_states = [sim.init_state.copy() for sim in self.agents.values()]
    #     # self.init_states[0].update({
    #     #     'ic_psi_true_deg': (self.np_random.uniform(270, 540))%360,
    #     #     'ic_h_sl_ft': self.np_random.uniform(17000, 23000),
    #     # })
    #     init_states = self.init_states.copy()
    #     self.np_random.shuffle(init_states)
    #     for idx, sim in enumerate(self.agents.values()):
    #         sim.reload(init_states[idx])
    #     self._tempsims.clear()
    def reset_simulators(self):
        # 首次保存每个 agent 的初始状态
        if self.init_states is None:
            self.init_states = [sim.init_state.copy() for sim in self.agents.values()]

        # 拷贝并打乱初始状态
        init_states = self.init_states.copy()
        self.np_random.shuffle(init_states)

        # 对每个 agent 的初始状态加扰动并加载#、
        for idx, sim in enumerate(self.agents.values()):
            state = init_states[idx].copy()

            # 基于原始值进行小幅扰动
            state['ic_psi_true_deg'] += float(self.np_random.uniform(-30, 30))  # 航向角扰动 ±30°
            state['ic_psi_true_deg'] %= 360  # 保持在 [0, 360) 范围内

            state['ic_h_sl_ft'] += float(self.np_random.uniform(-1000, 1000))  # 高度扰动 ±1000 英尺
            state['ic_long_gc_deg'] += float(self.np_random.uniform(-0.1, 0.1))  # 经度扰动 ±0.1°
            state['ic_u_fps'] += float(self.np_random.uniform(-50, 50))  # 初始速度扰动 ±50 fps

            sim.reload(state)

        self._tempsims.clear()

