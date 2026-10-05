"""
scripts/sap_probe_flow_positions.py
Diagnose the per-position latent MSE and token accuracy of ChunkMeanFlowPrior across the 240 chunk positions.
"""
import torch
import numpy as np
from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from scripts.sap_chunk_ae import ChunkAE
from scripts.sap_meanflow_chunk import ChunkMeanFlowPrior

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer("/vol/tokenizer_sap")
    V = tok.get_vocab_size()
    
    chunk_ae = ChunkAE(K=8, dz=256, V=V).to(device)
    ae_ckpt = torch.load("/vol/out/s03_sap/s13_chunk_ae_K8_dz256.pt", map_location=device, weights_only=True)
    chunk_ae.load_state_dict(ae_ckpt["state_dict"] if "state_dict" in ae_ckpt else ae_ckpt)
    chunk_ae.eval()

    prior = ChunkMeanFlowPrior(V, 128, 240, 256, 512, 8, 8).to(device)
    p_ckpt = torch.load("/vol/out/s03_sap/s13_meanflow_K8_dz256_d8.pt", map_location=device, weights_only=True)
    prior.load_state_dict(p_ckpt["state_dict"] if "state_dict" in p_ckpt else p_ckpt)
    prior.eval()

    loader = tokenizing_distributed_data_loader_bos_bestfit(tok, 128, 2048, split="val", data_dir="/vol/data", device=str(device))
    batch, _ = next(loader)
    prompt, target = batch[:, :128], batch[:, 128:128+240*8]
    
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_true = chunk_ae.encode(target.reshape(-1, 8)).reshape(128, 240, 256)
            z_pred = prior.sample_one_pass(prompt)
            logits = chunk_ae.decode(z_pred.reshape(-1, 256))
            pred_toks = logits.argmax(-1).reshape(128, 240, 8)
            target_toks = target.reshape(128, 240, 8)
            
        pos_mse = (z_pred.float() - z_true.float()).pow(2).mean(dim=(0, 2)).cpu().numpy()
        pos_acc = (pred_toks == target_toks).float().mean(dim=(0, 2)).cpu().numpy()

    print("=== PER-CHUNK ANALYSIS ===")
    print("Chunks 0-4 (tokens 0-32 after prompt):")
    for i in range(5):
        print(f"  Chunk {i:3d} (toks {i*8:4d}-{(i+1)*8:4d}): MSE = {pos_mse[i]:.4f}, Token Acc = {pos_acc[i]*100:.2f}%")

    print("\nChunks 10-14 (tokens 80-112 after prompt):")
    for i in range(10, 15):
        print(f"  Chunk {i:3d} (toks {i*8:4d}-{(i+1)*8:4d}): MSE = {pos_mse[i]:.4f}, Token Acc = {pos_acc[i]*100:.2f}%")

    print("\nChunks 115-120 (tokens ~1000):")
    for i in range(115, 120):
        print(f"  Chunk {i:3d} (toks {i*8:4d}-{(i+1)*8:4d}): MSE = {pos_mse[i]:.4f}, Token Acc = {pos_acc[i]*100:.2f}%")

    print("\nChunks 235-239 (tokens ~1900):")
    for i in range(235, 240):
        print(f"  Chunk {i:3d} (toks {i*8:4d}-{(i+1)*8:4d}): MSE = {pos_mse[i]:.4f}, Token Acc = {pos_acc[i]*100:.2f}%")

    print(f"\nAverage MSE: {pos_mse.mean():.4f}")
    print(f"Min MSE: {pos_mse.min():.4f} at chunk {pos_mse.argmin()}")
    print(f"Max MSE: {pos_mse.max():.4f} at chunk {pos_mse.argmax()}")

if __name__ == "__main__":
    main()
