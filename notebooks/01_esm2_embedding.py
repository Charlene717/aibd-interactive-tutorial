"""
01_esm2_embedding.py
====================
Compute ESM-2 embeddings for protein sequences and compare similarities.
Corresponds to: Ch01 (Hello AI×Bio) and Ch11 (Protein Structure).

Requirements
------------
- fair-esm
- torch (GPU recommended, but CPU works for tiny model)

Expected runtime: ~30 s on CPU, <5 s on GPU.

Usage
-----
$ python 01_esm2_embedding.py
"""
import torch
import esm


def load_model(name: str = "esm2_t6_8M_UR50D"):
    """Load a small ESM-2 model and its alphabet."""
    model, alphabet = getattr(esm.pretrained, name)()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


def embed_sequences(seqs: list[tuple[str, str]], model, alphabet, layer: int = 6):
    """Return (n, dim) mean-pooled embeddings for a list of (name, sequence)."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter(seqs)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        out = model(tokens, repr_layers=[layer])
    reps = out["representations"][layer]
    # Mean over tokens (skip the special <bos> at position 0)
    return reps[:, 1:, :].mean(dim=1)


def cosine_matrix(emb: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity matrix."""
    n = emb / emb.norm(dim=-1, keepdim=True)
    return n @ n.T


if __name__ == "__main__":
    model, alphabet = load_model("esm2_t6_8M_UR50D")
    proteins = [
        ("human_insulin_A", "GIVEQCCTSICSLYQLENYCN"),
        ("human_insulin_B", "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
        ("bovine_insulin_A", "GIVEQCCASVCSLYQLENYCN"),  # 1 aa diff
        ("random_short", "MKTAYIAKQRQISFVKSHFSRQ"),
    ]
    emb = embed_sequences(proteins, model, alphabet)
    sim = cosine_matrix(emb).cpu()
    names = [p[0] for p in proteins]
    print(f"\nEmbedding shape: {tuple(emb.shape)}")
    print("\nPairwise cosine similarity:")
    print(f"{'':<22}" + "".join(f"{n:>22}" for n in names))
    for i, n in enumerate(names):
        row = "".join(f"{sim[i, j].item():>22.4f}" for j in range(len(names)))
        print(f"{n:<22}{row}")

    print("\nExpect: human_insulin_A ~ bovine_insulin_A (very similar)\n"
          "        human_insulin_A vs random_short (dissimilar)")
