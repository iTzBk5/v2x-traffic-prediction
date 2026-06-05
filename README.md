# V2X Trend Prediction with PPO and Online Metaheuristic Guidance

Traffic direction prediction system for V2X (Vehicle-to-Everything) networks using deep reinforcement learning. The model predicts dominant traffic flow direction at road intersections, combining a PPO-trained actor-critic with online PSO/GA logit-bias optimization.

## Architecture

- **Per-RSU TCN Encoder**: Temporal Convolutional Network processes time-series data from each Road Side Unit
- **Multi-Head Attention**: Cross-RSU attention captures spatial dependencies between intersections
- **PPO Actor-Critic**: Proximal Policy Optimization trains the direction predictor as an RL agent
- **Online Metaheuristic Guide**: PSO and GA continuously adapt logit biases at inference time to handle distribution shift

## Project Structure

```
work/
├── arch/
│   └── top.py                  # Training pipeline (data generation, PPO, LSTM baseline, ARIMA, evaluation, plotting)
└── sumo_sim/
    ├── 01_build_network.py     # Downloads OSM data and builds SUMO road network
    ├── generate_3h_traffic.py  # Generates 3-hour traffic demand with rush hour patterns
    ├── 03_compare_simulations.py  # Side-by-side SUMO comparison (baseline vs AI-controlled TLS)
    ├── model_def.py            # Model architecture and OnlineMetaheuristicGuide (inference-only)
    └── v2x_gym/                # Gymnasium environment for direction prediction
        └── envs/
            └── direction_pred_env.py
```

## Methods Compared

| Method | Description |
|--------|-------------|
| PPO + PSO | PPO policy + Particle Swarm Optimization online bias |
| PPO + GA | PPO policy + Genetic Algorithm online bias |
| LSTM | Supervised bidirectional LSTM baseline |
| ARIMA | Statistical time-series forecasting |
| Kalman | Kalman filter with damped trend |
| Random | Uniform random baseline |

## Requirements

- Python 3.10+
- PyTorch
- NumPy, Matplotlib, OSMnx, Gymnasium
- SUMO (for simulation comparison)
- statsmodels (optional, for ARIMA)

## Usage

### Train and evaluate all methods
```bash
python work/arch/top.py
```

### Run SUMO simulation comparison
```bash
python work/sumo_sim/01_build_network.py
python work/sumo_sim/generate_3h_traffic.py
python work/sumo_sim/03_compare_simulations.py
```
