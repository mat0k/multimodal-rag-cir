import numpy as np
import torch
from PIL import Image

from src.datasets.cirr import build_cirr_dataset
from src.fusion import fusion
from src.retrievers.vista_retriever import VistaBGERetriever


model = VistaBGERetriever.from_pretrained(
    model_name_or_path="BAAI/bge-base-en-v1.5",
    checkpoint_path="models/Visualized_BGE/Visualized_base_en_v1.5.pth",
)
model.eval()

vision_device = next(model.vision.parameters()).device
text_device = next(model.text.parameters()).device

triplets = build_cirr_dataset(
    split="val",
    mode="triplets",
    image_transform=model.image_processor,
    caption_transform=model.tokenizer,
    max_length_tokenizer=77,
)
queries = [triplets[i] for i in [0, 1, 2]]

index_ds = build_cirr_dataset(
    split="val",
    mode="images",
    image_transform=model.image_processor,
    caption_transform=model.tokenizer,
    max_length_tokenizer=77,
)
all_names = list(index_ds.name_to_relpath.keys())
seed_candidates = all_names[:1000]
extra = []
for q in queries:
    extra.extend([q["reference_name"], q["target_name"]])
candidate_names = list(dict.fromkeys(seed_candidates + extra))

cand_feats = []
bs = 64
for s in range(0, len(candidate_names), bs):
    chunk = candidate_names[s : s + bs]
    tensors = []
    for name in chunk:
        image_path = "data/cirr/images/" + index_ds.name_to_relpath[name]
        image = Image.open(image_path).convert("RGB")
        tensor = model.image_processor(image, return_tensors="pt")["pixel_values"][0]
        tensors.append(tensor)
    batch = torch.stack(tensors, dim=0).to(vision_device)
    with torch.no_grad():
        feats = model.vision(batch).image_embeds
    cand_feats.append(feats)

cand_feats = torch.nn.functional.normalize(torch.cat(cand_feats, dim=0), dim=-1)
arr_names = np.array(candidate_names)

ref_images = torch.stack([q["reference"] for q in queries], dim=0).to(vision_device)
input_ids = torch.stack([q["transformed_caption"] for q in queries], dim=0).to(text_device)
attn = torch.stack([q["attention_mask"] for q in queries], dim=0).to(text_device)

with torch.no_grad():
    img_reps = model.vision(ref_images).image_embeds
    txt_reps = model.text(input_ids=input_ids, attention_mask=attn).text_embeds
    q_fused = fusion(img_reps, txt_reps, fusion_type="sum", alpha=0.7)

    q_mm = model.backbone.encode_mm(ref_images, {"input_ids": input_ids, "attention_mask": attn})
    q_mm = torch.nn.functional.normalize(q_mm, dim=-1)

s_fused = q_fused @ cand_feats.T
s_mm = q_mm @ cand_feats.T


def rank_without_reference(scores_row, ref_name):
    order = torch.argsort(scores_row, descending=True).cpu().numpy()
    names = arr_names[order]
    vals = scores_row[order].detach().cpu().numpy()
    mask = names != ref_name
    return names[mask], vals[mask]


print("CIRR ranking sanity sample on fixed 1,000-candidate gallery (queries idx=0,1,2)")
for i, q in enumerate(queries):
    pair_id = q["pair_id"]
    ref = q["reference_name"]
    tgt = q["target_name"]
    cap = q["caption"]

    n_f, v_f = rank_without_reference(s_fused[i], ref)
    n_m, v_m = rank_without_reference(s_mm[i], ref)

    pos_f = np.where(n_f == tgt)[0]
    pos_f = int(pos_f[0] + 1) if len(pos_f) else -1
    pos_m = np.where(n_m == tgt)[0]
    pos_m = int(pos_m[0] + 1) if len(pos_m) else -1

    overlap5 = len(set(n_f[:5]).intersection(set(n_m[:5])))

    print("-" * 100)
    print(f"pair_id={pair_id} target={tgt} reference={ref}")
    print(f"caption={cap}")
    print(f"fused target_rank={pos_f} | mm target_rank={pos_m} | top5_overlap={overlap5}/5")
    print("fused top5 ids:", ", ".join(n_f[:5].tolist()))
    print("fused top5 scores:", ", ".join([f"{x:.4f}" for x in v_f[:5]]))
    print("mm    top5 ids:", ", ".join(n_m[:5].tolist()))
    print("mm    top5 scores:", ", ".join([f"{x:.4f}" for x in v_m[:5]]))

print("-" * 100)
print("done")
