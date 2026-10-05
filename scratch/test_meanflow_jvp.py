import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class JVPTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim = heads, width // heads
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, -1)
        def heads(a):
            return a.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        att = (q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)).softmax(-1)
        mixed = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(mixed)
        return x + self.ff(self.ln2(x))

class ChunkMeanFlowPrior(nn.Module):
    def __init__(self, V_prompt: int, P: int, N: int, dz: int, width: int = 512, depth: int = 6, heads: int = 8):
        super().__init__()
        self.P = P  # prompt tokens (128)
        self.N = N  # chunk latents (240)
        self.dz = dz # chunk latent dim (256)
        self.width = width
        
        self.prompt_emb = nn.Embedding(V_prompt, width)
        self.prompt_pos = nn.Parameter(torch.randn(P, width) * 0.02)
        
        self.latent_in = nn.Linear(dz, width)
        self.latent_pos = nn.Parameter(torch.randn(N, width) * 0.02)
        
        self.time_emb = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )
        
        self.blocks = nn.ModuleList([JVPTransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, dz)

    def forward(self, z, r, t, prompt_tokens):
        # z: (B, N, dz)
        # r: (B,)
        # t: (B,)
        # prompt_tokens: (B, P)
        B = z.size(0)
        rt = torch.stack((r, t, t - r), -1) # (B, 3)
        t_vec = self.time_emb(rt)[:, None, :] # (B, 1, width)
        
        h_prompt = self.prompt_emb(prompt_tokens) + self.prompt_pos[None, :, :]
        h_latents = self.latent_in(z) + self.latent_pos[None, :, :] + t_vec
        
        # Concatenate prompt and chunk latents: (B, P + N, width)
        h = torch.cat([h_prompt, h_latents], dim=1)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        # Latent predictions from the N chunk positions
        out = self.out(h[:, self.P:, :])
        return out

    def loss(self, prompt_tokens, x_latents):
        # x_latents: (B, N, dz) clean chunk latents from frozen ChunkAE
        B, N, dz = x_latents.shape
        device = x_latents.device
        dtype = x_latents.dtype
        eps = torch.randn_like(x_latents)
        
        # Sample time intervals
        a = torch.randn(B, device=device).sub_(0.4).sigmoid()
        b = torch.randn(B, device=device).sub_(0.4).sigmoid()
        r, t = torch.minimum(a, b), torch.maximum(a, b)
        same = torch.rand_like(r) >= 0.25
        r = torch.where(same, t, r)
        
        # Linear trajectory
        # z_t = (1 - t) x + t eps
        t_expand = t[:, None, None]
        z_t = (1 - t_expand) * x_latents + t_expand * eps
        v = eps - x_latents
        
        fn = lambda zz, rr, tt: self.forward(zz, rr, tt, prompt_tokens)
        u, dudt = torch.func.jvp(fn, (z_t, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
        
        dt_expand = (t - r)[:, None, None]
        target = (v - dt_expand * dudt).detach()
        err = (u - target).square().mean((1, 2))
        weight = (err.detach() + 1e-3).rsqrt()
        loss = (weight * err).mean()
        return loss, {"flow_mse": float(err.mean().detach())}

    @torch.no_grad()
    def sample_one_pass(self, prompt_tokens):
        # One-step generation from pure Gaussian noise: r=0, t=1
        B = prompt_tokens.size(0)
        device = prompt_tokens.device
        eps = torch.randn(B, self.N, self.dz, device=device)
        r = torch.zeros(B, device=device)
        t = torch.ones(B, device=device)
        u = self.forward(eps, r, t, prompt_tokens)
        # x_hat = eps - u(eps, 0, 1)
        x_hat = eps - u
        # Unit RMS normalization as expected by ChunkAE
        x_hat = x_hat * torch.rsqrt(x_hat.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x_hat

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Testing on device:', device)
    V_prompt = 32000
    P = 128
    N = 240
    dz = 256
    model = ChunkMeanFlowPrior(V_prompt, P, N, dz, width=256, depth=2, heads=4).to(device)
    
    prompt = torch.randint(0, V_prompt, (2, P), device=device)
    x = torch.randn(2, N, dz, device=device)
    # Unit RMS
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    
    loss, metrics = model.loss(prompt, x)
    print('Loss computed successfully:', loss.item(), metrics)
    loss.backward()
    print('Backward pass successful!')
    
    sample = model.sample_one_pass(prompt)
    print('One-pass sample shape:', sample.shape, 'RMS:', sample.pow(2).mean().sqrt().item())

    # Test with autocast bfloat16
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, m = model.loss(prompt, x)
    loss.backward()
    opt.step()
    print('Autocast + backward + opt step successful! Loss:', loss.item())
