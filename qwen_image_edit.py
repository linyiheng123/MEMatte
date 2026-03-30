"""
Qwen Image Edit – combine MEMatte matting with the Qwen/DashScope image-editing
API to produce composited images with AI-generated or stylised backgrounds.

Usage examples
--------------
# Provide a pre-computed alpha matte (skip MEMatte):
python qwen_image_edit.py \
    --image  path/to/image.jpg \
    --alpha  path/to/alpha.png \
    --prompt "a snowy mountain landscape" \
    --output result.png

# Let MEMatte compute the alpha matte on the fly:
python qwen_image_edit.py \
    --image          path/to/image.jpg \
    --trimap         path/to/trimap.png \
    --config-dir     ./configs/MEMatte_S_topk0.25_win_global_long.py \
    --checkpoint-dir ./checkpoints/MEMatte_ViTS_DIM.pth \
    --prompt         "a tropical beach at sunset" \
    --output         result.png

The DashScope API key can be supplied via --api-key or the DASHSCOPE_API_KEY
environment variable.
"""

import os
import base64
import argparse
import urllib.request
from io import BytesIO

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URI."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def composite(fg: Image.Image, alpha: np.ndarray, bg: Image.Image) -> Image.Image:
    """Alpha-composite *fg* over *bg* using the float32 [0, 1] array *alpha*."""
    fg_rgb = np.array(fg.convert("RGB"), dtype=np.float32) / 255.0
    bg_rgb = np.array(bg.convert("RGB").resize(fg.size, Image.LANCZOS),
                      dtype=np.float32) / 255.0
    a = alpha[:, :, np.newaxis]
    out = fg_rgb * a + bg_rgb * (1.0 - a)
    return Image.fromarray((out.clip(0.0, 1.0) * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# MEMatte matting
# ---------------------------------------------------------------------------

def run_matting(
    image_path: str,
    trimap_path: str,
    config_dir: str,
    checkpoint_dir: str,
    max_number_token: int = 18500,
):
    """Run MEMatte on a single image and return (PIL RGB image, PIL L alpha)."""
    import torch
    from torchvision.transforms import functional as TF
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import LazyConfig, instantiate

    cfg = LazyConfig.load(config_dir)
    cfg.model.teacher_backbone = None
    cfg.model.backbone.max_number_token = max_number_token
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = instantiate(cfg.model)
    model.to(device)
    model.eval()
    DetectionCheckpointer(model).load(checkpoint_dir)

    image = Image.open(image_path).convert("RGB")
    trimap = Image.open(trimap_path).convert("L")

    img_t = TF.to_tensor(image).unsqueeze(0).to(device)
    tri_t = TF.to_tensor(trimap)[0:1].unsqueeze(0).to(device)

    with torch.no_grad():
        out, _, _ = model({"image": img_t, "trimap": tri_t}, patch_decoder=True)
        alpha_t = out["phas"].flatten(0, 2)
        tri_sq = tri_t.squeeze()
        alpha_t[tri_sq == 0] = 0.0
        alpha_t[tri_sq == 1] = 1.0

    alpha_pil = TF.to_pil_image(alpha_t.cpu())
    return image, alpha_pil


# ---------------------------------------------------------------------------
# Qwen / DashScope image editing
# ---------------------------------------------------------------------------

def generate_background(prompt: str, reference_image: Image.Image, api_key: str = None) -> Image.Image:
    """
    Generate a background image using the Qwen/DashScope image-synthesis API.

    The *reference_image* is passed so the model can match style/composition.
    Falls back to a plain text-to-image call when the API does not accept a
    reference image.

    Parameters
    ----------
    prompt : str
        Text description of the desired background.
    reference_image : Image.Image
        The original (foreground) image, used as a style reference.
    api_key : str, optional
        DashScope API key; falls back to DASHSCOPE_API_KEY env var.

    Returns
    -------
    Image.Image
        Generated background as an RGB PIL image.
    """
    try:
        import dashscope
        from dashscope import ImageSynthesis
    except ImportError as exc:
        raise ImportError(
            "The 'dashscope' package is required for Qwen image editing. "
            "Install it with:  pip install dashscope"
        ) from exc

    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError(
            "A DashScope API key is required. "
            "Pass --api-key or set the DASHSCOPE_API_KEY environment variable."
        )
    dashscope.api_key = key

    ref_uri = _encode_image_base64(reference_image)
    w, h = reference_image.size
    # DashScope accepts sizes as 'WxH' strings; clamp to valid values.
    size = _nearest_valid_size(w, h)

    response = ImageSynthesis.call(
        model="wanx-v1",
        prompt=prompt,
        ref_img=ref_uri,
        n=1,
        size=size,
    )

    if response.status_code == 200 and response.output.results:
        url = response.output.results[0].url
        with urllib.request.urlopen(url) as resp:
            return Image.open(BytesIO(resp.read())).convert("RGB")

    raise RuntimeError(
        f"DashScope API request failed (HTTP {response.status_code}): "
        f"{getattr(response, 'message', response)}"
    )


def _nearest_valid_size(w: int, h: int) -> str:
    """Return the closest supported DashScope image-synthesis size string."""
    valid = [
        (512, 512),
        (768, 512), (512, 768),
        (768, 768),
        (1024, 576), (576, 1024),
        (1024, 768), (768, 1024),
        (1024, 1024),
    ]
    best = min(valid, key=lambda s: abs(s[0] - w) + abs(s[1] - h))
    return f"{best[0]}x{best[1]}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Edit images with MEMatte matting + Qwen image editing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input image ---
    p.add_argument("--image", required=True, metavar="PATH",
                   help="Input RGB image (JPG or PNG)")

    # --- Alpha / matting ---
    alpha_group = p.add_argument_group("Alpha matte (choose one)")
    alpha_group.add_argument("--alpha", metavar="PATH",
                             help="Pre-computed alpha matte (PNG, single channel). "
                                  "If supplied, MEMatte is not run.")
    alpha_group.add_argument("--trimap", metavar="PATH",
                             help="Trimap (PNG) used by MEMatte to compute alpha.")
    alpha_group.add_argument("--config-dir", metavar="PATH",
                             help="MEMatte config file (.py).")
    alpha_group.add_argument("--checkpoint-dir", metavar="PATH",
                             help="MEMatte checkpoint (.pth).")
    alpha_group.add_argument("--max-number-token", type=int, default=18500,
                             metavar="N",
                             help="Max tokens for MEMatte patch inference "
                                  "(default: %(default)s)")

    # --- Qwen editing ---
    edit_group = p.add_argument_group("Qwen image editing")
    edit_group.add_argument("--prompt", required=True, metavar="TEXT",
                            help="Text prompt describing the desired background.")
    edit_group.add_argument("--api-key", metavar="KEY",
                            help="DashScope API key (or set DASHSCOPE_API_KEY).")

    # --- Output ---
    p.add_argument("--output", required=True, metavar="PATH",
                   help="Path to save the composited output image (PNG).")
    return p


def main():
    args = build_parser().parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load / compute alpha matte
    # ------------------------------------------------------------------ #
    if args.alpha:
        print(f"Loading pre-computed alpha from: {args.alpha}")
        image = Image.open(args.image).convert("RGB")
        alpha_pil = Image.open(args.alpha).convert("L")
    elif args.trimap and args.config_dir and args.checkpoint_dir:
        print("Running MEMatte to compute alpha matte …")
        image, alpha_pil = run_matting(
            image_path=args.image,
            trimap_path=args.trimap,
            config_dir=args.config_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_number_token=args.max_number_token,
        )
        print("Matting complete.")
    else:
        build_parser().error(
            "Provide either --alpha OR (--trimap + --config-dir + --checkpoint-dir)."
        )

    # ------------------------------------------------------------------ #
    # 2. Generate background via Qwen / DashScope
    # ------------------------------------------------------------------ #
    print(f"Generating background with prompt: {args.prompt!r} …")
    background = generate_background(
        prompt=args.prompt,
        reference_image=image,
        api_key=args.api_key,
    )
    print("Background generation complete.")

    # ------------------------------------------------------------------ #
    # 3. Composite foreground over new background
    # ------------------------------------------------------------------ #
    alpha_arr = np.array(alpha_pil, dtype=np.float32) / 255.0
    result = composite(image, alpha_arr, background)

    # ------------------------------------------------------------------ #
    # 4. Save
    # ------------------------------------------------------------------ #
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    result.save(args.output)
    print(f"Result saved to: {args.output}")


if __name__ == "__main__":
    main()
