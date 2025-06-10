import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from scipy.signal import butter, filtfilt
import numpy as np

# -------------------------------------------------------------
# 1. Feature Extractor (CBAM + Sub‑band depthwise + Variance)
# -------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=4):
        super().__init__()
        # 1D 풀링: (B, C, T) → (B, C, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # MLP: C → C/r → C
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio, bias=True),  # W0
            nn.ReLU(inplace=True),
            nn.Linear(gate_channels // reduction_ratio, gate_channels, bias=True),  # W1
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor of shape (B, C, T)
        returns: Tensor of shape (B, C, T)
        """
        b, c, t = x.size()
        
        avg = self.avg_pool(x).view(b, c)  # (B, C)
        max = self.max_pool(x).view(b, c)  # (B, C)
        
        avg_out = self.mlp(avg)           # (B, C)
        max_out = self.mlp(max)           # (B, C)
        
        scale = self.sig(avg_out + max_out)      # (B, C)
        
        scale = scale.view(b, c, 1)       # (B, C, 1)
        return x * scale                  # (B, C, T) 브로드캐스트 곱
        
class SpatialAttention(nn.Module):
    """
    1D 입력 (B, C, T) 에서 채널 축 풀링 후 1×1 Conv1d 으로
    공간 어텐션 맵 M_s(f)를 계산합니다. (n=1)
    
    수식 (2): M_s(f) = σ(f_{1×1}([f^s_avg; f^s_max]))
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: Tensor, shape (B, C, T)
        returns: Tensor, same shape (B, C, T)
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        concat = torch.cat([avg_out, max_out], dim=1)
        
        scale = self.sigmoid(self.conv(concat))
        return x * scale

class CBAM(nn.Module):
	def __init__(self, gate_channels, reduction_ratio=16):
		super(CBAM, self).__init__()
		self.ChannelGate = ChannelAttention(gate_channels, reduction_ratio)
		self.SpatialGate = SpatialAttention()

	def forward(self, x):
		x = self.ChannelGate(x)
		x = self.SpatialGate(x)
		return x

class FFTBandpass(nn.Module):
    def __init__(self, subbands, fs, T):
        """
        subbands: list of (fl, fh)
        fs: 샘플링 주파수
        T: 입력 시퀀스 길이 (예: 1000)
        """
        super().__init__()
        self.subbands = subbands
        self.fs = fs
        # 1) 주파수 축 계산
        freq = torch.fft.rfftfreq(T, d=1/fs)  # (T//2+1,)
        masks = []
        for fl, fh in subbands:
            # 각 서브밴드에 대한 마스크
            m = ((freq >= fl) & (freq <= fh)).float()
            masks.append(m)
        # (m, F) 형태로 뭉치고 buffer 로 등록
        self.register_buffer('masks', torch.stack(masks, dim=0))  # (m, F)

    def forward(self, x):
        """
        x: Tensor, shape (B, C, T)
        returns: Tensor, shape (B, m, C, T)
        """
        # 2) FFT
        X = torch.fft.rfft(x, dim=-1)               # (B, C, F)
        outs = []
        for m in self.masks:                        # m: (F,)
            # 채널·배치에 브로드캐스트
            Xm = X * m.unsqueeze(0).unsqueeze(0)    # (B, C, F)
            # 3) IFFT
            x_bp = torch.fft.irfft(Xm, n=x.size(-1), dim=-1)  # (B, C, T)
            outs.append(x_bp)
        # 4) 서브밴드를 첫번째 dim 으로 쌓기
        return torch.stack(outs, dim=1)             # (B, m, C, T)

class DepthwiseConv(nn.Module):
    """
    입력:  x.shape = (B, m, C, T)    # m = 밴드 수 (여기서는 입력 채널)
    출력:  y.shape = (B, d, m, 1, T)
    """
    def __init__(self, bands: int, depth_multiplier: int, num_electrodes: int):
        super().__init__()
        self.bands = bands               # = in_channels
        self.depth = depth_multiplier   # = depth_multiplier
        # depthwise conv2d: out_channels = in_channels * depth_multiplier
        # groups = in_channels 으로 해야 각 채널마다 별도 필터 적용
        self.dw = nn.Conv2d(
            in_channels=bands,
            out_channels=bands * depth_multiplier,
            kernel_size=(num_electrodes, 1),
            groups=bands,
            bias=False
        )
        self.bn_dw = nn.BatchNorm2d(bands * depth_multiplier)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, m, C, T)
        B, m, C, T = x.shape
        # 1) 바로 depthwise conv: (B, m, C, T) → (B, m*d, 1, T)
        y = self.dw(x)
        y = self.bn_dw(y)                       # BatchNorm 적용
        # 2) 채널(axis=1) 차원 m*d 를 (m, d) 로 분리
        y = self.activation(y)                  # 활성화 함수
        y = self.dropout(y)                     # 드롭아웃
        # y = y.view(B, m, self.depth, 1, T)      # (B, m, d, 1, T)
        # # 3) 축 순서 바꿔서 (B, d, m, 1, T) 로
        # y = y.permute(0, 2, 1, 3, 4)
        return y

# class VarianceLayer(nn.Module):
#     def __init__(self, w):
#         super().__init__()
#         self.w = w
#     def forward(self, x):   # (B,m,d,1,T)
#         *front, T = x.shape
#         pad_len = (-T) % self.w
#         if pad_len > 0:
#             x = F.pad(x, (0, pad_len), mode='constant', value=0.0)
#         new_T = x.size(-1)
#         num_win = new_T // self.w
        
#         x = x.view(*front, num_win, self.w)
        
#         v = x.var(dim=-1, unbiased=False)
#         return torch.clamp(v, min=1e-6)

class VarianceLayer(nn.Module):
    """
    Variance Layer: computes variance over non-overlapping temporal windows.

    Input:
        x of shape (B, C, 1, T)
          - B: batch size
          - C: 채널 수 (e.g., m * d)
          - 1: spatial 차원 (이미 합쳐진 상태)
          - T: 전체 타임포인트 수

    Output:
        y of shape (B, C, 1, K), where K = T // w
    """
    def __init__(self, w: int):
        super(VarianceLayer, self).__init__()
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 차원 정리
        #   x: (B, C, 1, T) → squeeze하여 (B, C, T)
        B, C, _, T = x.shape
        x = x.view(B, C, T)

        # 2) 창 개수 계산 및 불필요한 마지막 부분 자르기
        K = T // self.w
        x = x[:, :, :K * self.w]          # (B, C, K*w)

        # 3) (B, C, K, w) 형태로 뷰 변경
        x = x.view(B, C, K, self.w)

        # 4) 시간평균 µ(k) 계산: (B, C, K, 1)
        mu = x.mean(dim=-1, keepdim=True)

        # 5) 분산 계산: (B, C, K)
        var = ((x - mu) ** 2).mean(dim=-1)

        # 6) 출력 형태 맞추기: (B, C, 1, K)
        y = var.unsqueeze(2)
        return y


class FeatureExtractor(nn.Module):
    def __init__(self, C=22, subbands=((8,12),(12,16),(16,20),(20,24),(24,28),(28,32)),
                 depth=22, var_win=250, fs=250, fc_out=64):
        super().__init__()
        self.fs = fs
        self.subbands = subbands
        
        # self.band_filters = nn.ModuleList()
        # numtaps = 101                                # 필터 길이 (홀수 권장)
        # for fl, fh in self.subbands:
        #     # normalized cutoff
        #     nyq = 0.5 * self.fs
        #     taps = firwin(numtaps,
        #                   [fl/nyq, fh/nyq],
        #                   pass_zero=False)
        #     # 2) depthwise Conv1d (groups=C) 으로 필터링
        #     conv = nn.Conv1d(in_channels=C,
        #                      out_channels=C,
        #                      kernel_size=numtaps,
        #                      groups=C,
        #                      bias=False,
        #                      padding=numtaps//2)
        #     # weight shape: (C, 1, numtaps)
        #     conv.weight.data.copy_(
        #         torch.tensor(taps, dtype=torch.float32)
        #              .view(1,1,-1)
        #              .repeat(C,1,1)
        #     )
        #     conv.weight.requires_grad = False   # 고정필터; 학습가능하게 두려면 True로
        #     self.band_filters.append(conv)
        self.fft_bp = FFTBandpass(self.subbands, fs=self.fs, T=1000)
        
        self.depth_convs = DepthwiseConv(bands=len(subbands), depth_multiplier=depth, num_electrodes=C)
        self.cbams = CBAM(gate_channels=C)
        self.var = VarianceLayer(var_win)
        self.fc = nn.Linear(depth * len(subbands) * (1000 // var_win), fc_out)

    def forward(self, x):   # x: (B, C, T)
        B, C, T = x.shape
        x = self.cbams(x)
        
        # for conv in self.band_filters:
        #     bp = conv(x)     # (B, C, T) —> same shape
        #     feats.append(bp)
        # feats = torch.stack(feats, dim=1)           # → (B, m, C, T)
        
        feats = self.fft_bp(x)         # (B, m, C, T)
        
        feats = self.depth_convs(feats)             # (B, d, m, 1, T)
        out = self.var(feats)                     # (B, d, m, 1, T//w)

        # out = out.flatten(1)                # (B, d*m*(T//w))
        # out  = self.fc(out)                 # (B, fc_out=64)
        return out.flatten(1)                    # Flatten to (B, total_features)

# -------------------------------------------------------------
# 2. Domain Discriminator (Critic)
# -------------------------------------------------------------
# class Critic(nn.Module):
#     def __init__(self, in_dim, hid=2048):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim,hid),
#             nn.LeakyReLU(.2,True),
#             nn.Linear(hid,hid),
#             nn.LeakyReLU(.2,True),
#             nn.Linear(hid,1)
#             )
#     def forward(self,h):
#         return self.net(h).view(-1)
class Critic(nn.Module):
    def __init__(self, in_dim, hid=[512,256]):
        super().__init__()
        layers = []
        dims = [in_dim] + hid
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_d, out_d),
                       nn.LeakyReLU(0.2, True)]
        layers += [nn.Linear(hid[-1], 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, h):
        return self.net(h).view(-1, 1)


# Gradient penalty
def gradient_penalty(critic, h_s, h_t, lambda_gp):
    alpha = torch.rand(h_s.size(0), 1, device=h_s.device)
    h_hat = alpha * h_s + (1 - alpha) * h_t
    h_hat.requires_grad_(True)
    d_hat = critic(h_hat)
    grad_out = torch.ones_like(d_hat)
    grad_h = grad(d_hat, h_hat, grad_out, create_graph=True, retain_graph=True)[0]
    return lambda_gp * ((grad_h.norm(2, 1) - 1) ** 2).mean()

# -------------------------------------------------------------
# 3. Classifier (2×FC)
# -------------------------------------------------------------
class Classifier(nn.Module):
    def __init__(self,in_dim, n_cls, hidden=512):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden)
        self.fc2=nn.Linear(hidden,n_cls)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(p=0.6)
    def forward(self,h):
        h = self.fc1(h)
        # h = self.bn1(h)
        # h = F.relu(h)
        # h = self.dropout(h)
        return F.softmax(self.fc2(h), dim=1)

# -------------------------------------------------------------
# 4. DANet wrapper
# -------------------------------------------------------------
class DANet(nn.Module):
    def __init__(self, C=22, n_cls=4, citic_hid=512, classifier_hidden=64, feat_dim=None):
        super().__init__()
        self.F = FeatureExtractor(C=C)
        if feat_dim is None:
            # 만일 누락되면 안전장치
            sample = torch.zeros(1, C, 1000)
            feat_dim = self.F(sample).shape[1]
        # self.D = Critic(feat_dim, hid=citic_hid)
        self.D = Critic(feat_dim)
        self.C = Classifier(feat_dim, n_cls, hidden=classifier_hidden)
    def forward(self, x):
        h = self.F(x)
        return h, self.D(h), self.C(h)

# -------------------------------------------------------------
# 5. single training iteration
# -------------------------------------------------------------
def train_iter(model, src_x, src_y, tgt_x, opts,
               lambda_gp=10, mu=1.0, critic_steps=5):

    Fnet, Clf, Cri = model.module.F, model.module.C, model.module.D
    opt_fc, opt_c, opt_d = opts
    device = next(Fnet.parameters()).device
    src_x, src_y, tgt_x = src_x.to(device), src_y.to(device), tgt_x.to(device)
    # print(src_x.shape, src_y.shape, tgt_x.shape)
    # exit()
    model.eval()
    model.module.D.train()
    wd_critic_list, wd_feat_list = [], []
    # ── (i) critic update ─────────────────────────────
    for _ in range(critic_steps):
        # 특징은 gradient 차단
        with torch.no_grad():
            h_s, _, _ = model(src_x)   # 여러 GPU에서 분산 계산
            h_t, _, _ = model(tgt_x)
            h_s = h_s.detach()
            h_t = h_t.detach()

        wd_critic = model.module.D(h_s).mean() - model.module.D(h_t).mean()
        gp = gradient_penalty(Cri, h_s, h_t, lambda_gp)
        loss_d = -(wd_critic) + gp          # gradient ASCENT
        
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        
        wd_critic_list.append(wd_critic.item())   # Critic 직후의 wd 기록
    wd_critic_mean = float(np.mean(wd_critic_list))
    
    # ── (ii) feature + classifier update ─────────────
    model.train()
    model.module.D.eval()
    
    h_s, _, logit = model(src_x)
    loss_c = F.cross_entropy(logit, src_y)
    loss_c_copy = loss_c.detach().clone()  # retain_graph=True 필요
    opt_c.zero_grad()
    loss_c.backward(retain_graph=True)  # retain_graph=True 필요
    opt_c.step()
    
    h_t, _, _ = model(tgt_x)
    wd_feat = model.module.D(h_s).mean() - model.module.D(h_t).mean()

    loss_f = (wd_feat * mu) + loss_c_copy
    
    opt_fc.zero_grad()
    loss_f.backward()
    opt_fc.step()
    wd_feat_list.append(wd_feat.item())   # Feature 단계 직후의 wd 기록

    return {
        'loss_D': loss_d.item(),
        'wd_critic': wd_critic_mean,
        'wd_feat': wd_feat.item(),
        'cls_loss': loss_c.item(),
        'loss_f': loss_f.item()
    }

def main():
    B, C, T = 4, 22, 1000
    x = torch.randn(B, C, T)

    # 1) CBAM 모듈 (4D 더미 입력)
    feat = CBAM(gate_channels=22)(x)
    print("ChannelAttention:", ChannelAttention(gate_channels=22)(x).shape)
    print("SpatialAttention:", SpatialAttention()(x).shape)
    print("CBAM:", feat.shape)

    # 2) Feature Extractor
    feat_extractor = FeatureExtractor(C=C)
    feat = feat_extractor(x)
    print("Feature Extractor output:", feat.shape)  # (B, 64)
    # 3) Critic / Classifier
    critic = Critic(in_dim=feat.shape[1])
    print("Critic output:", critic(feat).shape)      # (B,)
    clf = Classifier(feat.shape[1], n_cls=4)
    print("Classifier output:", clf(feat).shape)     # (B,4)

    # 4) DANet end-to-end
    net = DANet(C=C, n_cls=4, feat_dim=feat.shape[1])
    feat2 = net.F(x)
    print("DANet Feature:", feat2.shape)
    print("DANet Critic:", net.D(feat2).shape)
    print("DANet Classifier:", net.C(feat2).shape)

if __name__ == "__main__":
    main()