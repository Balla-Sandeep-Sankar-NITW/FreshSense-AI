# main.py — CLI / script entry point for FreshSense AI

import argparse
import logging
import sys

import cv2

from agent.agent import FreshSenseAgent
from recipes.recommender import recommend
from utils.image_utils import pil_to_bgr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FreshSenseAI")


def main():
    parser = argparse.ArgumentParser(description="FreshSense AI — CLI")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--day",  type=int, default=0, help="Current global day")
    parser.add_argument("--save", default="", help="Save annotated image to path")
    args = parser.parse_args()

    # load image
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot read image: {args.image}")
        sys.exit(1)

    agent = FreshSenseAgent()
    agent.global_day = args.day

    output = agent.process(img)

    print(f"\n=== FreshSense AI — Day {output['global_day']} ===\n")

    for g in output["results"]:
        print(
            f"  {g['fruit_type']:15s} ×{g['count']}  "
            f"day={g['predicted_day']:2d}  "
            f"rem={g['remaining_days']:+d}d  "
            f"[{g['alert']}]  "
            f"conf={g['avg_confidence']:.2f}"
        )

    print("\n--- Events ---")
    for e in output["events"]:
        print(" ", e)

    recs = recommend(output["results"])
    if recs:
        print("\n--- Recipe Recommendations ---")
        for r in recs:
            print(f"  {r['emoji']} {r['name']}  ({', '.join(r['matched'])})  [{r['urgency']}]")

    print("\n--- Learned Shelf Lives ---")
    for k, v in sorted(output["learned_shelf"].items()):
        print(f"  {k:15s}: {v:.1f} days")

    if args.save:
        cv2.imwrite(args.save, output["annotated"])
        print(f"\nAnnotated image saved to: {args.save}")


if __name__ == "__main__":
    main()
