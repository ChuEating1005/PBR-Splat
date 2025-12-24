#!/usr/bin/env python3
"""
Create MP4 videos from rendered frame sequences.
Requires ffmpeg to be installed.
"""

import os
import subprocess
from pathlib import Path
from argparse import ArgumentParser


def create_video(frames_dir, output_path, fps=30, quality="high"):
    """
    Create video from image sequence using ffmpeg.
    
    Args:
        frames_dir: Directory containing sequential images (00000.png, 00001.png, ...)
        output_path: Output video path (.mp4)
        fps: Frames per second
        quality: Video quality ("high", "medium", "low")
    """
    if not os.path.exists(frames_dir):
        print(f"❌ Error: Frames directory not found: {frames_dir}")
        return False
    
    # Check if frames exist
    frame_pattern = os.path.join(frames_dir, "%05d.png")
    first_frame = os.path.join(frames_dir, "00000.png")
    
    if not os.path.exists(first_frame):
        print(f"❌ Error: No frames found in {frames_dir}")
        return False
    
    # Quality presets
    crf_values = {
        "high": 18,    # Higher quality, larger file
        "medium": 23,  # Balanced
        "low": 28      # Lower quality, smaller file
    }
    crf = crf_values.get(quality, 23)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path
    ]
    
    print(f"Creating video: {output_path}")
    print(f"  Source: {frames_dir}")
    print(f"  FPS: {fps}")
    print(f"  Quality: {quality} (CRF={crf})")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Video created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating video:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Error: ffmpeg not found. Please install ffmpeg first.")
        print("   Ubuntu/Debian: sudo apt install ffmpeg")
        print("   macOS: brew install ffmpeg")
        return False


def create_combined_video(input_dir, output_path, fps=30, quality="high", types=None):
    """
    Create a combined video with all G-buffer types stacked horizontally.
    
    Args:
        input_dir: Directory containing individual G-buffer videos
        output_path: Output video path (.mp4)
        fps: Frames per second (should match individual videos)
        quality: Video quality ("high", "medium", "low")
        types: List of G-buffer types to combine
    """
    if types is None:
        types = ["albedo", "normal", "metallic", "roughness"]
    
    input_dir = Path(input_dir)
    
    # Check if all required videos exist
    video_paths = []
    for gtype in types:
        video_path = input_dir / f"{gtype}.mp4"
        if not video_path.exists():
            print(f"❌ Error: Required video not found: {video_path}")
            print(f"   Please create individual videos first.")
            return False
        video_paths.append(str(video_path))
    
    # Quality presets
    crf_values = {
        "high": 18,    # Higher quality, larger file
        "medium": 23,  # Balanced
        "low": 28      # Lower quality, smaller file
    }
    crf = crf_values.get(quality, 23)
    
    # Build ffmpeg command with horizontal stacking
    # Create filter: [0:v][1:v][2:v][3:v]hstack=inputs=4
    num_inputs = len(types)
    filter_inputs = "".join(f"[{i}:v]" for i in range(num_inputs))
    filter_complex = f"{filter_inputs}hstack=inputs={num_inputs}"
    
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if exists
    ]
    
    # Add input files
    for video_path in video_paths:
        cmd.extend(["-i", video_path])
    
    # Add filter and output options
    cmd.extend([
        "-filter_complex", filter_complex,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path)
    ])
    
    print(f"\nCreating combined video: {output_path}")
    print(f"  Combining: {' + '.join(types)}")
    print(f"  Layout: Horizontal stack")
    print(f"  Quality: {quality} (CRF={crf})")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Combined video created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating combined video:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("❌ Error: ffmpeg not found. Please install ffmpeg first.")
        return False


def main():
    parser = ArgumentParser(description="Create videos from rendered frame sequences")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing G-buffer subdirectories (albedo, normal, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for videos (default: same as input_dir)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second")
    parser.add_argument("--quality", type=str, default="high",
                        choices=["high", "medium", "low"],
                        help="Video quality")
    parser.add_argument("--types", type=str, nargs="+",
                        default=["albedo", "normal", "metallic", "roughness"],
                        help="G-buffer types to create videos for")
    parser.add_argument("--create_combined", action="store_true",
                        help="Also create a combined video with all G-buffers stacked horizontally")
    parser.add_argument("--combined_name", type=str, default="combined",
                        help="Name for the combined video file (default: combined.mp4)")
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        return
    
    print(f"Creating videos from: {input_path}")
    print(f"Output directory: {output_path}\n")
    
    # Create video for each G-buffer type
    success_count = 0
    for gtype in args.types:
        frames_dir = input_path / gtype
        video_path = output_path / f"{gtype}.mp4"
        
        if not frames_dir.exists():
            print(f"⚠️  Skipping {gtype}: directory not found")
            continue
        
        if create_video(str(frames_dir), str(video_path), args.fps, args.quality):
            success_count += 1
        print()
    
    print(f"✅ Created {success_count}/{len(args.types)} videos")
    
    # Create combined video if requested
    if args.create_combined and success_count == len(args.types):
        combined_output = output_path / f"{args.combined_name}.mp4"
        print("\n" + "="*60)
        if create_combined_video(output_path, combined_output, args.fps, args.quality, args.types):
            print(f"\n🎬 All videos created successfully!")
            print(f"   Individual videos: {output_path}/")
            print(f"   Combined video: {combined_output}")
        print("="*60)
    elif args.create_combined and success_count < len(args.types):
        print(f"\n⚠️  Skipping combined video creation: not all individual videos were created successfully.")


if __name__ == "__main__":
    main()

