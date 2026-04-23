# PPO Experiments (Independent Folder)

这个目录是**独立**的 PPO 实现，不会修改或依赖你前面的 DQN 训练代码路径。

## 文件说明

- `ppo_train.py`：单次训练入口（离散动作 PPO，支持 `CartPole-v0` / `Acrobot-v1`）
- `run_ppo_experiments.py`：按环境和种子批量运行
- `summarize_ppo_results.py`：汇总多 seed 结果并绘图
- `requirements.txt`：本目录依赖说明

## 快速开始

```bash
cd /root/project/project_1_code/ppo_code
python ppo_train.py --env CartPole-v0 --seed 0 --results-dir ppo_results
python ppo_train.py --env Acrobot-v1 --seed 0 --results-dir ppo_results
```

## 批量运行

```bash
cd /root/project/project_1_code/ppo_code
python run_ppo_experiments.py --env CartPole-v0 --seeds 0 1 2 --results-dir ppo_comp_result
python run_ppo_experiments.py --env Acrobot-v1 --seeds 0 1 2 --results-dir ppo_comp_result
```

## 结果汇总

```bash
cd /root/project/project_1_code/ppo_code
python summarize_ppo_results.py --results-dir ppo_comp_result --envs CartPole-v0 Acrobot-v1
```

## 输出结构

每个 run 在 `results-dir/<env>_seed<seed>/` 下产出：

- `log.csv`
- `model.pt`
- `training_rewards.png`
- `eval_rewards.png`
- `loss.png`

汇总图在 `results-dir/summary/` 下。
