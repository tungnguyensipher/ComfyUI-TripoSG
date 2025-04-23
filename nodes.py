import os
import cv2
import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

import folder_paths
import comfy.model_management as mm
from comfy_extras.nodes_hunyuan3d import MESH

# Import pipeline classes
from .triposg.pipelines.pipeline_triposg import TripoSGPipeline
from .triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline


gpu = mm.get_torch_device()
cpu = torch.device("cpu")


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def simplify_mesh(mesh: MESH, n_faces: int):
    # Assume mesh.vertices: (1, N, 3), mesh.faces: (1, M, 3)
    v = mesh.vertices[0].cpu().numpy()
    f = mesh.faces[0].cpu().numpy()

    if f.shape[0] <= n_faces or n_faces == 0:
        # No simplification needed, just return original
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        try:
            import pymeshlab
        except ImportError:
            raise ImportError("pymeshlab is not installed. Please install it with `pip install pymeshlab`.")
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        m = ms.current_mesh()
        vertices = torch.from_numpy(m.vertex_matrix()).float().unsqueeze(0)
        faces = torch.from_numpy(m.face_matrix()).long().unsqueeze(0)
    return MESH(vertices=vertices, faces=faces)


class TripoSGModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": (["VAST-AI/TripoSG", "VAST-AI/TripoSG-scribble"], {"default": "VAST-AI/TripoSG"})}
        }

    RETURN_TYPES = ("TRIPOSG",)
    FUNCTION = "load_model"
    CATEGORY = "TripoSG"

    def load_model(self, model):
        model_name = model.split("/")[-1]
        model_dir = os.path.join(folder_paths.models_dir, "3D", model_name)
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"Downloading {model} to {model_dir}")
            snapshot_download(repo_id=model, local_dir=model_dir, local_dir_use_symlinks=False)

        if model == "VAST-AI/TripoSG":
            pipe = TripoSGPipeline.from_pretrained(model_dir).to(gpu, torch.float16)
        elif model == "VAST-AI/TripoSG-scribble":
            pipe = TripoSGScribblePipeline.from_pretrained(model_dir).to(gpu, torch.float16)
        else:
            raise ValueError(f"Unknown model: {model}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return (pipe,)


class TripoSGInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRIPOSG",),
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
            },
            "optional": {
                "conditioning": ("TRIPOSG_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "run_inference"
    CATEGORY = "TripoSG"

    def run_inference(
        self,
        model,
        image,
        seed,
        steps,
        cfg,
        conditioning={},
    ):
        pil_image = tensor2pil(image)

        pipe_class = model.__class__.__name__
        generator = torch.Generator(device=model.device).manual_seed(seed)

        if pipe_class == "TripoSGPipeline":
            outputs = model(
                image=pil_image,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
            ).samples[0]
        elif pipe_class == "TripoSGScribblePipeline":
            prompt = conditioning.get("prompt", "")
            if not prompt:
                raise ValueError("Prompt is required for TripoSGScribblePipeline")

            outputs = model(
                image=pil_image,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=0,  # CFG-distilled model
                use_flash_decoder=False,
                dense_octree_depth=8,
                hierarchical_octree_depth=8,
                **conditioning,
            ).samples[0]
        else:
            raise ValueError(f"Unknown pipeline type: {pipe_class}")

        # Convert outputs to torch tensors and add batch dimension
        vertices = (
            torch.from_numpy(outputs[0].astype(np.float32))
            if not isinstance(outputs[0], torch.Tensor)
            else outputs[0].float()
        )
        faces = (
            torch.from_numpy(np.ascontiguousarray(outputs[1]))
            if not isinstance(outputs[1], torch.Tensor)
            else outputs[1]
        )
        vertices = vertices.unsqueeze(0)  # shape (1, N, 3)
        faces = faces.unsqueeze(0)  # shape (1, M, 3)
        mesh = MESH(vertices, faces)

        return (mesh,)


class SimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "faces": (
                    "INT",
                    {
                        "min": 0.0,
                        "max": 0xFFFFFFFFFFFFFFF,
                        "step": 1,
                        "default": 0,
                        "tooltip": "The number of faces to simplify the mesh to. 0 means no simplification.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "simplify_mesh"
    CATEGORY = "TripoSG"

    def simplify_mesh(self, mesh, faces):
        if faces == 0 or faces > mesh.faces.shape[0]:
            return (mesh,)

        return (simplify_mesh(mesh, faces),)


class TripoSGPrepareImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prepare"
    CATEGORY = "TripoSG"

    def prepare(self, image, mask=None):
        # image: [1, H, W, C] or [H, W, C], float32, 0-1
        # mask: [1, H, W] or [H, W], float32, 0-1 or 0-255
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Image tensor must be [H, W, C], got {image.shape}")
        H, W, C = image.shape
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        alpha = None

        # Handle channels
        if C == 1:
            rgb_image = np.repeat(image_np, 3, axis=2)  # HWC
        elif C == 3:
            rgb_image = image_np  # HWC
        elif C == 4:
            rgb_image = image_np[:, :, :3]  # HWC
            alpha = image_np[:, :, 3]
        else:
            raise ValueError(f"Unsupported channel count: {C}")

        # Resize if too large
        H, W = rgb_image.shape[:2]
        max_side = max(H, W)
        if max_side > 2048:
            scale = 2048 / max_side
            new_H, new_W = int(H * scale), int(W * scale)
            rgb_image = cv2.resize(rgb_image, (new_W, new_H), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            H, W = new_H, new_W

        # Alpha validation
        def is_valid_alpha(alpha, min_ratio=0.01):
            hist = cv2.calcHist([alpha], [0], None, [20], [0, 256])
            min_hist_val = alpha.shape[0] * alpha.shape[1] * min_ratio
            return hist[0] >= min_hist_val and hist[-1] >= min_hist_val

        if alpha is not None and not is_valid_alpha(alpha):
            alpha = None

        if alpha is None and mask is None:
            raise ValueError("Image has no valid alpha channel, please provide a mask.")

        if alpha is None:
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape != (H, W):
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {(H, W)}")
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            alpha = mask_np

        # Find bounding box
        if np.any(alpha > 0):
            x, y, w, h = self.find_bounding_box(alpha)
        else:
            raise ValueError("input image too small or empty mask")

        # Compose with white background
        alpha_f = alpha.astype(np.float32) / 255.0
        rgb_f = rgb_image.astype(np.float32) / 255.0
        bg_color = np.ones(3, dtype=np.float32)  # [1,1,1]
        out_rgb = rgb_f * alpha_f[..., None] + bg_color * (1 - alpha_f[..., None])

        # Crop to bbox
        cropped = out_rgb[y : y + h, x : x + w, :]

        # Pad to square with 10% padding
        pad_ratio = 0.1
        pad_h = int(max(h, w) * pad_ratio)
        size = max(h, w) + 2 * pad_h
        padded = np.ones((size, size, 3), dtype=np.float32)

        # Center crop
        y_off = (size - h) // 2
        x_off = (size - w) // 2
        padded[y_off : y_off + h, x_off : x_off + w, :] = cropped

        # To tensor [1, H, W, 3]
        tensor = torch.from_numpy(padded).unsqueeze(0).contiguous().float()
        return (tensor,)

    @staticmethod
    def find_bounding_box(gray_image):
        # gray_image: HxW uint8
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, gray_image.shape[1], gray_image.shape[0]
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, w, h


class TripoSGConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "prompt_confidence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scribble_confidence": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("TRIPOSG_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "conditioning"
    CATEGORY = "TripoSG"

    def conditioning(self, prompt, prompt_confidence, scribble_confidence):
        return (
            {
                "prompt": prompt,
                "attention_kwargs": {
                    "cross_attention_scale": prompt_confidence,
                    "cross_attention_2_scale": scribble_confidence,
                },
            },
        )


# Node registration
NODE_CLASS_MAPPINGS = {
    "TripoSGModelLoader": TripoSGModelLoader,
    "TripoSGInference": TripoSGInference,
    "TripoSGConditioning": TripoSGConditioning,
    "SimplifyMesh": SimplifyMesh,
    "TripoSGPrepareImage": TripoSGPrepareImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSGModelLoader": "TripoSG Model Loader",
    "TripoSGInference": "TripoSG Inference",
    "TripoSGConditioning": "TripoSG Conditioning",
    "SimplifyMesh": "Simplify Mesh",
    "TripoSGPrepareImage": "TripoSG Prepare Image",
}
