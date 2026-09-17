import torch
import torch.nn as nn

EMBD = 16            # embedding dim per modality
WINDOW = 16          # timesteps in the sliding window
N_ACTIONS = 20       # total number of joints
N_BINS = 11          # discrete bins per joint

# input dimensions per modality
MODALITY_DIMS = {
    "asymmetric": 25,
    "goal": 4,
    "proprioception": 48,
    "touch": 92,
    "vision": 7,
}

MODALITIES = list(MODALITY_DIMS.keys())     
N_MOD = len(MODALITIES)                       

class FrozenKVMemory(nn.Module):
    """
    Frozen key/value memory
    """
    def __init__(self, embd=EMBD, head_size=EMBD, decay=0.95):
        super().__init__()
        self.key   = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        # freeze KV memory
        for p in self.key.parameters():
            p.requires_grad = False
        for p in self.value.parameters():
            p.requires_grad = False
        self.decay = decay

    def forward(self, x_emb, q):
        B, T, _ = x_emb.shape
        k = self.key(x_emb)      # (B, T, HEAD)  
        v = self.value(x_emb)    # (B, T, HEAD)  
        Dk = k.shape[-1]
        Dv = v.shape[-1]
        lam = self.decay

        # kv-memory
        M = torch.zeros(B, Dv, Dk, device=x_emb.device, dtype=x_emb.dtype)
        outs = []
        for t in range(T):
            term = v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2) # (B, Dv, Dk)
            M = lam * M + term
            qt = q[:, t].unsqueeze(-1)                           # (B, Dk, 1)
            outs.append((M @ qt).squeeze(-1))                    # (B, Dv)
        return torch.stack(outs, dim=1)                          # (B, T, Dv)


class MotorAttentionNet(nn.Module):
    """
    Full model.
    """
    def __init__(self, embd=EMBD, head_size=EMBD, decay=0.95, use_motor=True):
        super().__init__()
        self.use_motor = use_motor

        # trainable embedding per sensory modality and motor stream
        self.embed = nn.ModuleDict({
            m: nn.Linear(dim, embd, bias=False) for m, dim in MODALITY_DIMS.items()
        })
        if use_motor:
            self.embed["motor"] = nn.Linear(N_ACTIONS, embd, bias=False)

        # frozen KV memory per stream 
        self.mem = nn.ModuleDict({
            m: FrozenKVMemory(embd, head_size, decay) for m in MODALITIES
        })
        if use_motor:
            self.mem["motor"] = FrozenKVMemory(embd, head_size, decay)

        # trainable query projection, driven by the motor embedding
        self.query = nn.Linear(embd, head_size, bias=False)
        # project the summed memory readout to joint class logits
        self.readout = nn.Linear(head_size, N_ACTIONS * N_BINS)

        # slices to split the concatenated 176-wide input back into modalities
        self.slices = {}
        start = 0
        for name, dim in MODALITY_DIMS.items():
            self.slices[name] = (start, start + dim)
            start += dim

    def forward(self, x, prev_pred=None):
        # x         : (B, T, 176)  sensor window
        # prev_pred : (B, T, 20)  previous-action window
        if prev_pred is None:
            prev_pred = x.new_zeros(x.shape[0], x.shape[1], N_ACTIONS)

        # motor embedding drives the query and its own memory
        if self.use_motor:
            motor_emb = self.embed["motor"](prev_pred)     # (B, T, embd)
            q = self.query(motor_emb)                      # (B, T, HEAD)
        else:
            # no motor stream: query from a zero embedding
            q = self.query(x.new_zeros(x.shape[0], x.shape[1], self.query.in_features))

        reads = []
        for m in MODALITIES:
            a, b = self.slices[m]
            emb_m = self.embed[m](x[..., a:b])     # (B, T, embd)
            y_m = self.mem[m](emb_m, q)            # (B, T, HEAD)
            reads.append(y_m[:, -1])               # separate readout, last timestep (B, HEAD)

        if self.use_motor:
            y_motor = self.mem["motor"](motor_emb, q)   # memory of the motor sequence
            reads.append(y_motor[:, -1])

        h = torch.stack(reads, dim=0).sum(dim=0)   # sum over memories -> (B, HEAD)
        logits = self.readout(h)                   # (B, N_ACTIONS * N_BINS)
        return logits.view(-1, N_ACTIONS, N_BINS)  # (B, 20, 11) per-joint logits


# data generator
def make_prev_pred_windows(y_full, window=WINDOW):
    import numpy as np
    N = y_full.shape[0]
    y_prev = np.zeros_like(y_full)
    y_prev[1:] = y_full[:-1]
    M = N - window + 1
    return np.stack([y_prev[i:i + window] for i in range(M)], axis=0)


# Autoregressive inference 
@torch.no_grad()
def autoregressive_predict(model, x_windows, window=WINDOW):
    """
    Inference: feed the model's previous predictions back in, step by step.
    """
    model.eval()
    M = x_windows.shape[0]
    preds = torch.zeros(M, N_ACTIONS, dtype=x_windows.dtype, device=x_windows.device)
    for i in range(M):
        xi = x_windows[i:i + 1]                      
        prev = torch.zeros(1, window, N_ACTIONS, dtype=xi.dtype, device=xi.device)
        for t in range(window):
            src = i + t - 1 - (window - 1)          
            if src >= 0:
                prev[0, t] = preds[src]
        logits = model(xi, prev)[0]                  # (N_ACTIONS, N_BINS)
        preds[i] = logits.argmax(dim=-1).to(preds.dtype)   # class ids
    return preds


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    x = torch.randn(B, WINDOW, 176)               # dummy sensor windows
    prev = torch.randn(B, WINDOW, N_ACTIONS)      # dummy previous-action windows

    net = MotorAttentionNet()
    out = net(x, prev)
    print("input :", tuple(x.shape))              # (4, 16, 176)
    print("prev  :", tuple(prev.shape))           # (4, 16, 20)
    print("logits:", tuple(out.shape))            # (4, 20, 11) per-joint logits
    assert out.shape == (B, N_ACTIONS, N_BINS)
    # softmax over the 11 bins
    probs = torch.softmax(out, dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(B, N_ACTIONS), atol=1e-5)

    # sensory memories and motor memory
    assert len(net.mem) == N_MOD + 1

    # frozen key and value ; trainable = embeddings, query, readout
    frozen = [n for n, p in net.named_parameters() if not p.requires_grad]
    train  = [n for n, p in net.named_parameters() if p.requires_grad]
    assert all((".key." in n or ".value." in n) for n in frozen)
    assert any("embed" in n for n in train)
    assert any("query" in n for n in train)
    assert any("readout" in n for n in train)
    print(f"frozen params: {len(frozen)} (key/value), trainable groups: embed/query/readout")
    print("OK: embed -> frozen KV memory (sensory + motor) -> query -> action")
