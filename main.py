import argparse
import os

import torch

from retrievers.base import BaseRetriever

import kiui
import numpy as np
from pipeline_mvrag import MVRAGPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Run MV-RAG")

    ## paths
    parser.add_argument('--ckpt_path', type=str, default="mvrag", help='Path to model checkpoint')
    parser.add_argument('--outputs_path', type=str, default="mvrag_outputs", help='Where to save output images')

    ## generation args
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for generation')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of DDIM diffusion steps')
    parser.add_argument('--num_initial_steps', type=int, default=50, help='Number of steps for fusion coefficient prediction')
    parser.add_argument('--seed', type=int, default=None, help='Seed for latents initialization')
    parser.add_argument('--elevation', type=int, default=15, help='How elevated the object is (in degrees)')
    parser.add_argument('--azimuth_start', type=int, default=0, help='Starting azimuth for generation (in degrees). E.g., choose 0 for direct views, 45 for diagonal views')

    ## retrieval args
    parser.add_argument('--k', type=int, default=4, help='Number of retrieved images')
    parser.add_argument('--retriever', type=str, default="simple", choices=["simple", "google", "bm25"], help='Retriever type. Choose simple for determined images input')
    # Simple retriever args
    parser.add_argument('--folder_path', type=str, help='Path to local folder containing images')
    # Google retriever args
    parser.add_argument('--google_api_key', type=str, help='Google API key')
    parser.add_argument('--google_cx', type=str, help='Google Search Engine ID')
    # bm25 args
    parser.add_argument('--index_path', type=str, help='Path for Lucene index')
    parser.add_argument('--image_base_path', type=str, help='Path for image base corresponding to the index')

    return parser.parse_args()


def get_retriever(args) -> BaseRetriever:
    if args.retriever == "simple":
        from retrievers.local_folder_retriever import SimpleRetriever
        if not args.folder_path:
            raise ValueError("Simple retriever requires --folder_path")
        return SimpleRetriever(args.folder_path)

    elif args.retriever == "bm25":
        from retrievers.bm25_retriever import BM25Retriever
        if not args.index_path or not args.image_base_path:
            raise ValueError("BM25 retriever requires --index_path and --image_base_path")
        return BM25Retriever(args.index_path, args.image_base_path)

    elif args.retriever == "google":
        from retrievers.google_images_retriever import GoogleRetriever
        if not args.google_api_key or not args.google_cx:
            raise ValueError("Google retriever requires --google_api_key and --google_cx")
        return GoogleRetriever(api_key=args.google_api_key, cx_id=args.google_cx)

    else:
        raise ValueError(f"Unsupported retriever type: {args.retriever}")


def main():
    args = parse_args()
    retriever = get_retriever(args)
    retrieved_images = retriever.retrieve(args.prompt, args.k)

    pipe = MVRAGPipeline.from_pretrained(
        args.ckpt_path,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    image = pipe(args.prompt,
                 retrieved_images,
                 guidance_scale=5,
                 num_inference_steps=args.num_inference_steps,
                 num_initial_steps=args.num_initial_steps,
                 elevation=args.elevation,
                 azimuth_start=args.azimuth_start,
                 seed=args.seed
                 )
    grid = np.concatenate(
        [
            np.concatenate([image[0], image[2]], axis=0),
            np.concatenate([image[1], image[3]], axis=0),
        ],
        axis=1,
    )
    output_path = os.path.join(args.outputs_path, f"{args.prompt}.png")
    kiui.write_image(output_path, grid)


if __name__ == "__main__":
    main()
