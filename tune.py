# tune.py
import optuna
import json
import wandb
from moabb_train import train_subject   # train_subject가 정의된 모듈
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import torch

# 공통 설정
dataset = BNCI2014_001()
eeg22 = [
    'Fz','FC3','FC1','FCz','FC2','FC4',
    'C5','C3','C1','Cz','C2','C4','C6',
    'CP3','CP1','CPz','CP2','CP4',
    'P1','Pz','P2','POz'
]
paradigm = MotorImagery(
    n_classes=4,
    channels=eeg22,
    fmin=8, fmax=32,
    tmin=2, tmax=6,
    resample=250
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # 1) 탐색할 하이퍼파라미터 정의
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-4)
    lambda_gp = trial.suggest_int("lambda_gp", 1, 20)
    critic_steps = trial.suggest_int("critic_steps", 5, 10)
    mu = trial.suggest_float("mu", 0.1, 5.0)
    cri_hid = trial.suggest_categorical("cri_hid", [128, 256, 512, 1024])
    cls_hid = trial.suggest_categorical("cls_hid", [128, 256, 512, 1024])
    epochs = 500  # 탐색 단계에선 짧게 잡는 것이 일반적
    batch = 256
    
    config = {
        "lr": lr,
        "lambda_gp": lambda_gp,
        "critic_steps": critic_steps,
        "mu": mu,
        "cri_hid": cri_hid,
        "cls_hid": cls_hid,
        "epochs": epochs,
        "batch": batch,
    }
    
    wandb.init(project="danet-optuna", config=config, reinit=True)

    # 2) LOO(subject A01)에 대해 성능 측정
    subj = "A01"
    acc, kappa = train_subject(
        subj=subj,
        dataset=dataset,
        paradigm=paradigm,
        epochs=epochs,
        batch=batch,
        lr=lr,
        gp=lambda_gp,
        mu=mu,
        critic_steps=critic_steps,
        cri_hid=cri_hid,
        cls_hid=cls_hid,
        device=device,
    )
    wandb.log({"accuracy": acc, "kappa": kappa})
    wandb.finish()
    # Optuna는 값이 클수록 좋은 방향으로 설정
    return acc  # 또는 kappa

if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (accuracy): {trial.value}")
    print("  Params:")
    for key, val in trial.params.items():
        print(f"    {key}: {val}")

    # 결과를 JSON으로 저장
    with open("optuna_best_params.json", "w") as f:
        json.dump({
            "accuracy": trial.value,
            "params": trial.params
        }, f, indent=2)
