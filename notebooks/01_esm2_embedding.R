## 01_esm2_embedding.R
## ====================
## R version of ESM-2 protein embedding via reticulate.
## Corresponds to: Ch01 / Ch11.
##
## Requirements: reticulate, plus Python `fair-esm` and `torch`.
## Expected runtime: ~30 s CPU.

library(reticulate)

esm   <- import("esm")
torch <- import("torch")

# Load small ESM-2
pre   <- esm$pretrained$esm2_t6_8M_UR50D()
model <- pre[[1]]; alphabet <- pre[[2]]
if (torch$cuda$is_available()) model <- model$cuda()

bc <- alphabet$get_batch_converter()

proteins <- list(
  reticulate::tuple("human_insulin_A",  "GIVEQCCTSICSLYQLENYCN"),
  reticulate::tuple("human_insulin_B",  "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"),
  reticulate::tuple("bovine_insulin_A", "GIVEQCCASVCSLYQLENYCN"),
  reticulate::tuple("random_short",     "MKTAYIAKQRQISFVKSHFSRQ")
)
res <- bc(proteins)
tokens <- if (torch$cuda$is_available()) res[[3]]$cuda() else res[[3]]

py_run_string("import torch")
out  <- model(tokens, repr_layers = list(6L))
reps <- out$representations[[6L]]
emb  <- reps[, 2:-1L, ]$mean(dim = 1L)   # mean-pool

# cosine
n <- emb / emb$norm(dim = -1L, keepdim = TRUE)
sim <- n$matmul(n$t())
sim_r <- py_to_r(sim$detach()$cpu()$numpy())
rownames(sim_r) <- colnames(sim_r) <- sapply(proteins, function(p) p[[1]])

cat("\nPairwise cosine similarity:\n")
print(round(sim_r, 4))
