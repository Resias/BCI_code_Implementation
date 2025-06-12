#!/usr/bin/env python
import os
import argparse, json
import numpy as np
import random

import torch
import torch.nn as nn 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import cohen_kappa_score
from scipy.signal import butter, filtfilt
from tqdm import trange, tqdm 

from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery

from model import FeatureExtractor, Critic, Classifier, DANet, train_iter
from itertools import cycle

import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# 캐시 디렉토리
CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU

    # 연산 일관성 확보 (성능 ↓ 가능)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────── DSP 유틸 ───────
def butter_bandpass(x: np.ndarray, low: float, high: float, fs: int, order: int = 4):
    """논문과 동일한 4차 Butterworth band-pass"""
    b, a = butter(order, [low/(0.5*fs), high/(0.5*fs)], btype="band")
    return filtfilt(b, a, x, axis=-1)

def exp_moving_standardize(x: np.ndarray, alpha=0.001, eps=1e-4):
    """논문에서 쓰인 exponential moving standardization"""
    # x shape: (n_trials, n_channels, n_times)
    out = np.empty_like(x, dtype=np.float32)
    # 채널마다 독립적으로 적용
    for trial in range(x.shape[0]):
        mean = np.zeros(x.shape[1])
        var  = np.ones(x.shape[1])
        for t in range(x.shape[2]):
            v = x[trial,:,t]
            mean = (1-alpha)*mean + alpha*v
            var  = (1-alpha)*var  + alpha*(v-mean)**2
            out[trial,:,t] = (v - mean)/np.sqrt(var + eps)
    return out

def butter_bandpass_batch(X: np.ndarray, subbands, fs: int = 250):
    n_trials, C, T = X.shape
    X_filt = []
    for (fl, fh) in subbands:
        b, a = butter(4, [fl / (0.5 * fs), fh / (0.5 * fs)], btype="band")
        X_band = filtfilt(b, a, X, axis=-1)
        X_filt.append(X_band)
    X_stack = np.stack(X_filt, axis=1)  # shape: (n_trials, n_subbands, C, T)
    return X_stack.astype(np.float32)


# ─────── 데이터 로드 & 전처리 ───────
def load_LOO_data(dataset, paradigm, target_subj):
    # MOABB subjects are 1–9
    all_subjs = list(range(1,10))
    t = int(target_subj[1:])           # "A03"→3
    src_subjs = [s for s in all_subjs if s!=t]
    # (1) source: train session만
    Xs_all, ys_all, meta_s = paradigm.get_data(dataset=dataset, subjects=src_subjs)
    mask_s = meta_s['session'] == '0train'
    Xs, ys = Xs_all[mask_s.values], ys_all[mask_s.values]
    # (2) target: test session만
    Xt_all, yt_all, meta_t = paradigm.get_data(dataset=dataset, subjects=[t])
    mask_t = meta_t['session'] == '1test'
    Xt, yt = Xt_all[mask_t.values], yt_all[mask_t.values]
    
    Xs = Xs[:, :, :-1]
    Xt = Xt[:, :, :-1]

    # (3) 논문과 똑같이 2–6 s 구간 segment가 이미 되어 있으므로
    #     MOABB가 반환한 Xs, Xt는 shape=(n_trials, n_chan, n_times)
    #     단, MOABB의 필터가 FIR-based라 다를 수 있으므로
    #     논문처럼 Butterworth로 다시 필터링하고 standardize
    fs = 250
    subbands = [(8,12),(12,16),(16,20),(20,24),(24,28),(28,32)]
    Xs = butter_bandpass(Xs, 8, 32, fs)
    Xt = butter_bandpass(Xt, 8, 32, fs)
    Xs = exp_moving_standardize(Xs)
    Xt = exp_moving_standardize(Xt)
    # Xs = exp_moving_standardize(Xs.reshape(-1, Xs.shape[2], Xs.shape[3])).reshape(Xs.shape)
    # Xt = exp_moving_standardize(Xt.reshape(-1, Xt.shape[2], Xt.shape[3])).reshape(Xt.shape)
    # (4) label을 0부터 시작하도록 리매핑
    classes = np.unique(ys)
    cls2idx = {c: i for i,c in enumerate(classes)}

    ys = np.vectorize(cls2idx.get)(ys)
    yt = np.vectorize(cls2idx.get)(yt)

    # torch tensor로 변환
    return (
        torch.tensor(Xs, dtype=torch.float32),
        torch.tensor(ys, dtype=torch.long)
    ), (
        torch.tensor(Xt, dtype=torch.float32),
        torch.tensor(yt, dtype=torch.long)
    )

# ─────── 이하 기존 코드 ───────
def make_loader(X, y=None, batch_size=64, train=True):
    ds = TensorDataset(X) if y is None else TensorDataset(X,y)
    if train:
        sampler = RandomSampler(ds, replacement=True,
                                num_samples=batch_size * (len(ds) //batch_size + len(ds) % batch_size))
        return DataLoader(ds, batch_size=batch_size,
                          sampler=sampler, drop_last=False)
    else:
        return DataLoader(ds, batch_size=batch_size,
                          shuffle=False, drop_last=False)

def evaluate(model, loader):
    model.eval()
    Ys, Ps = [], []
    device = next(model.module.F.parameters()).device
    with torch.no_grad():
        for batch in loader:
            X = batch[0].to(device)
            _, _, pred = model(X)
            p_ = pred.argmax(dim=1)
            Ps.append(p_.cpu())
            Ys.append(batch[1].cpu())

    y = torch.cat(Ys).numpy()
    p = torch.cat(Ps).numpy()
    # print(f" - Prediction logits: {p[:10]}")
    # print(f" - Prediction (argmax): {y[:10]}")
    acc = (y==p).mean()
    kappa = cohen_kappa_score(y,p)
    return acc, kappa

def load_LOO_data_cached(dataset, paradigm, target_subj):
    """
    캐시된 파일이 있으면 불러오고, 없으면 생성 후 저장합니다.
    저장 파일명: cache/{subj}_2a.pt
    """
    cache_path = os.path.join(CACHE_DIR, f"{target_subj}_2a.pt")
    if os.path.exists(cache_path):
        # 캐시 로드
        data = torch.load(cache_path)
        return data["Xs"], data["ys"], data["Xt"], data["yt"]
    
    # 캐시가 없으면 원래 load_LOO_data 로직 수행
    (Xs, ys), (Xt, yt) = load_LOO_data(dataset, paradigm, target_subj)

    # 캐시에 저장
    torch.save({
        "Xs": Xs,
        "ys": ys,
        "Xt": Xt,
        "yt": yt
    }, cache_path)

    return Xs, ys, Xt, yt

def train_subject(subj, dataset, paradigm, epochs, batch, lr, gp, mu, critic_steps, depth, cri_hid, cls_hid, device, wb):
    Xs, ys, Xt, yt = load_LOO_data_cached(dataset, paradigm, subj)
    print(f"Train (source) dataset length: {len(Xs)}")
    print(f"Validation/Test (target) dataset length: {len(Xt)}")
    src_loader  = make_loader(Xs, ys, batch, train=True)
    # tgt_loader  = make_loader(Xt, None, batch, train=True)
    num_src_batches = len(src_loader)
    tgt_sampler = RandomSampler(
        TensorDataset(Xt),
        replacement=True,
        num_samples=num_src_batches * batch
    )
    tgt_loader = DataLoader(
        TensorDataset(Xt),
        batch_size=batch,
        sampler=tgt_sampler,
        drop_last=False
    )
    test_loader = make_loader(Xt, yt, batch, train=False)

    C = Xs.shape[1]  # channels = 22
    T = Xs.shape[2]  # time points = 1000
    n_cls = int(ys.max().item()+1)
    n_cls_t = int(yt.max().item()+1)
    print(f"=== Subject {subj} | C={C}, T={T}, classes={n_cls} === | n_cls_t={n_cls_t}")

    tmpF = FeatureExtractor(C=C, depth=depth).to(device)
    d = torch.zeros(4, C, T).to(device)
    feat_dim = tmpF(d).shape[1]

    model = DANet(C=C, depth=depth, n_cls=n_cls, citic_hid=cri_hid, classifier_hidden=cls_hid, feat_dim=feat_dim)
    ngpu = torch.cuda.device_count()

    model = nn.DataParallel(model, device_ids=list(range(ngpu)))
    model = model.cuda()
    
    # weight_decay = 5e-4
    opt_fc = torch.optim.Adam(list(model.module.F.parameters()) + list(model.module.C.parameters()), lr=lr[1])
    # opt_c = torch.optim.Adam(list(model.module.C.parameters()), lr=lr[1])
    opt_d = torch.optim.Adam(list(model.module.D.parameters()), lr=lr[0], betas=(0.5, 0.9))

    scheduler_fc = torch.optim.lr_scheduler.StepLR(opt_fc, step_size=50, gamma=0.5)
    scheduler_d  = torch.optim.lr_scheduler.StepLR(opt_d,  step_size=50, gamma=0.5)
    
    best_acc = 0.0
    best_loss_f = 999.9
    best_loss_d = 0.0
    # Early stopping 설정
    patience = 200             # 개선이 없을 때 최대 허용 Epoch 수
    epochs_no_improve = 0
    for ep in trange(1, epochs+1, desc=f"{subj} Training"):
        # tgt_iter = cycle(tgt_loader)  # 무한 순환 타겟 배치
        # train_pairs = zip(src_loader, tgt_loader)
        # epoch마다 새로운 랜덤 시퀀스 iterator 생성
        tgt_iter = iter(tgt_loader)
        batch_bar = tqdm(
            src_loader,
            # train_pairs,
            desc=f"Epoch {ep} (source batches)",
            leave=False,
        )
        
        sum_lossD = 0.0
        sum_lossF = 0.0
        sum_cls = 0.0
        sum_wd_critic = 0.0
        sum_wd_feat = 0.0
        n_batches = 0
        
        for Xs_b, ys_b in batch_bar:
            Xt_b, = next(tgt_iter)  # 반복적으로 target 배치 꺼냄
        # for (Xs_b, ys_b), (Xt_b,) in batch_bar:
            stats = train_iter(
                model,
                Xs_b, ys_b, Xt_b,
                # [opt_fc, opt_c, opt_d],
                [opt_fc, opt_d],
                lambda_gp=gp, mu=mu, critic_steps=critic_steps)
            # set_postfix로 stats 출력
            batch_bar.set_postfix(
                lossD=f"{stats['loss_D']:.5f}",
                cls=f"{stats['cls_loss']:.5f}",
                wd_critic=f"{stats['wd_critic']:.5f}",
                wd_feat=f"{stats['wd_feat']:.5f}"
            )
            sum_lossD += stats['loss_D']
            sum_cls   += stats['cls_loss']
            sum_wd_critic += stats['wd_critic']
            sum_wd_feat   += stats['wd_feat']
            sum_lossF   += stats['loss_f']
            n_batches += 1
        
        avg_lossD     = sum_lossD / n_batches
        avg_cls       = sum_cls / n_batches
        avg_wd_critic = sum_wd_critic / n_batches
        avg_wd_feat   = sum_wd_feat / n_batches
        avg_lossF   = sum_lossF / n_batches
        
        if wb:
            wandb.log({
                "epoch": ep,
                "loss_D": avg_lossD,
                "cls_loss": avg_cls,
                "wd_critic": avg_wd_critic,
                "wd_feat": avg_wd_feat,
                "loss_F": avg_lossF
            })
        # scheduler_fc.step()
        # scheduler_d.step()
        acc, kappa = evaluate(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            best_loss_d = avg_lossD
            print(f"New best accuracy: {best_acc:.4f} (epoch {ep})")
        if wb:
            wandb.log({
                "epoch": ep,
                "test/acc": acc,
                "test/kappa": kappa
            })
        # ——— Early Stopping 체크 ———
        if best_loss_f > avg_lossF:
            best_loss_f = avg_lossF
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # 개선 정체 시 종료
        if epochs_no_improve >= patience:
            print(f"\nNo improvement for {patience} epochs (best_acc={best_acc:.4f}), stopping early.")
            break

    return best_acc, kappa

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--cls_lr", type=float, default=5e-4)
    p.add_argument("--lambda_gp", type=int, default=10)
    p.add_argument("--critic_steps", type=int, default=5)
    p.add_argument("--mu", type=float, default=1)
    p.add_argument("--cri_hid", type=int, default=64)
    p.add_argument("--depth_multiplier", type=int, default=2)
    p.add_argument("--cls_hid", type=int, default=64)
    p.add_argument("--wandb", type=bool, default=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    # cbam reduction ration 16으로 돌렸었음, 4확인해야함
    
    # 1) device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    seed = 42
    set_random_seed(seed)
    
    eeg22 = [
        'Fz','FC3','FC1','FCz','FC2','FC4',
        'C5','C3','C1','Cz','C2','C4','C6',
        'CP3','CP1','CPz','CP2','CP4',
        'P1','Pz','P2','POz'
    ]
    dataset = BNCI2014_001()
    paradigm = MotorImagery(
        n_classes=4,
        channels=eeg22,
        tmin=2, tmax=6,
        # resample=250
    )

    results = {}
    for i in range(1,10):
        subj = f"A{i:02d}"
        if args.wandb:
            wandb.init(project="danet-eeg-0610", name=subj, config={
                "epochs": args.epochs,
                "batch_size": args.batch,
                "critic learning_rate": args.critic_lr,
                "cls learning_rate": args.cls_lr,
                "lambda_gp": args.lambda_gp,
                "mu": args.mu,
                "critic_steps": args.critic_steps,
                # "critic_hidden": args.cri_hid,
                "classifier_hidden": args.cls_hid,
                # "seed": seed
            })
        acc, κ = train_subject(
            subj, dataset, paradigm,
            args.epochs, args.batch, [args.critic_lr, args.cls_lr],
            args.lambda_gp, args.mu, args.critic_steps,
            args.depth_multiplier, args.cri_hid, args.cls_hid, device, args.wandb
        )
        results[subj] = {"accuracy":acc, "kappa":κ}
        if args.wandb:
            wandb.run.summary["best_acc"] = acc
            wandb.run.summary["best_kappa"] = κ
            wandb.finish()

    print("\n=== Final LOO Results ===")
    with open("result.json", "w") as json_file:
        json.dump(results, json_file, indent=2)
    print(json.dumps(results, indent=2))
    