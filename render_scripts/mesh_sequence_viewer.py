#!/usr/bin/env python3
"""
Mesh Sequence Viewer using Polyscope

This script loads a sequence of mesh files from a folder and displays them
as an animation using Polyscope. It supports common mesh formats like OBJ,
PLY, OFF, and STL.

Features:
    - Interactive mesh sequence viewing and animation
    - Two-step video export: screenshots + ffmpeg video creation
    - Screenshots saved to [folder]/screenshots/ directory
    - High-quality MP4 video output using ffmpeg
    - Automatic exclusion of intermediate files (intxxx_xxxx.ply pattern)
    - Smooth animation controls with customizable frame rate

Usage:
    python mesh_sequence_viewer.py [folder_path]

Controls:
    - Spacebar: Play/Pause animation
    - Left/Right arrows: Previous/Next frame
    - R: Reset to first frame
    - 1-9: Set animation speed (1=slowest, 9=fastest)

Video Export Process:
    1. Click "Export Screenshots" - creates frame_*.png files
    2. Click "Create Video" - uses ffmpeg to make MP4

Output:
    - Screenshots: [input_folder]/screenshots/frame_000000.png, frame_000001.png, ...
    - Video: [input_folder]/[folder_name]_sequence.mp4

Dependencies:
    brew install ffmpeg  # For video creation (macOS)
    # or download from https://ffmpeg.org/download.html
"""

import os
import sys
import glob
import time
import argparse
import re
import tempfile
import shutil
import subprocess
from pathlib import Path
import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim
except ImportError:
    print("Error: polyscope is not installed. Install it using:")
    print("pip install polyscope")
    sys.exit(1)

try:
    import igl
    HAS_IGL = True
except ImportError:
    HAS_IGL = False
    print("Warning: libigl is not available. Only basic mesh loading will be supported.")
    print("Install it using: pip install libigl")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh is not available. Limited mesh format support.")
    print("Install it using: pip install trimesh")

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: opencv-python is not available. Video export will not be supported.")
    print("Install it using: pip install opencv-python")


class MeshSequenceViewer:
    def __init__(self, folder_path, end_frame=None):
        self.folder_path = Path(folder_path)
        self.mesh_files = []
        self.meshes = []
        self.current_frame = 0
        self.playing = False
        self.last_update_time = time.time()
        self.frame_rate = 10.0  # FPS
        self.mesh_name = "mesh_sequence"
        self.end_frame = end_frame
        
        # Load mesh files
        self._load_mesh_files()
        if not self.mesh_files:
            raise ValueError(f"No mesh files found in {folder_path}")
        
        # Load meshes
        self._load_meshes()
        if not self.meshes:
            raise ValueError("Failed to load any meshes")
        
        print(f"Loaded {len(self.meshes)} meshes from {folder_path}")
    
    def _should_exclude_file(self, filepath):
        """Check if a file should be excluded based on naming patterns."""
        filename = os.path.basename(filepath)
        # Exclude files matching pattern intxxx_xxxx.ply (intermediate results)
        if re.match(r'^int.*_.*\.ply$', filename):
            return True
        return False
    
    def _load_mesh_files(self):
        """Find all mesh files in the folder and sort them."""
        supported_extensions = ['*.obj', '*.ply', '*.off', '*.stl', '*.mesh']
        
        for ext in supported_extensions:
            pattern = str(self.folder_path / ext)
            files = glob.glob(pattern, recursive=False)
            # Filter out excluded files
            filtered_files = [f for f in files if not self._should_exclude_file(f)]
            self.mesh_files.extend(filtered_files)
        
        # Also check for files without extension or other extensions
        all_files = glob.glob(str(self.folder_path / "*"))
        for file in all_files:
            if os.path.isfile(file) and file not in self.mesh_files:
                # Skip excluded files
                if self._should_exclude_file(file):
                    continue
                # Try to determine if it's a mesh file by attempting to load it
                if self._is_mesh_file(file):
                    self.mesh_files.append(file)
        
        # Sort files naturally (handle numbers in filenames correctly)
        self.mesh_files.sort(key=self._natural_sort_key)
        print(f"Found {len(self.mesh_files)} mesh files")
    
    def _natural_sort_key(self, filename):
        """Natural sorting key for filenames with numbers."""
        import re
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        return [convert(c) for c in re.split('([0-9]+)', os.path.basename(filename))]
    
    def _is_mesh_file(self, filepath):
        """Check if a file might be a mesh file by trying to load it."""
        try:
            if HAS_IGL:
                V, F = igl.read_triangle_mesh(filepath)
                return V.shape[0] > 0 and F.shape[0] > 0
            elif HAS_TRIMESH:
                mesh = trimesh.load(filepath)
                return hasattr(mesh, 'vertices') and len(mesh.vertices) > 0
            else:
                return False
        except:
            return False
    
    def _load_meshes(self):
        """Load all meshes from the files."""
        print("Loading meshes...")
        for i, filepath in enumerate(self.mesh_files):
            try:
                vertices, faces = self._load_single_mesh(filepath)
                if vertices is not None and faces is not None:
                    self.meshes.append((vertices, faces))
                    if (i + 1) % 10 == 0:
                        print(f"Loaded {i + 1}/{len(self.mesh_files)} meshes...")
                else:
                    print(f"Failed to load mesh: {filepath}")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
            
            if self.end_frame is not None and i + 1 >= self.end_frame:
                break
        
        print(f"Successfully loaded {len(self.meshes)} meshes")
    
    def _load_single_mesh(self, filepath):
        """Load a single mesh file."""
        if HAS_IGL:
            try:
                V, F = igl.read_triangle_mesh(filepath)
                if V.shape[0] > 0 and F.shape[0] > 0:
                    return V.astype(np.float64), F.astype(np.int32)
            except:
                pass
        
        if HAS_TRIMESH:
            try:
                mesh = trimesh.load(filepath)
                if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                    return np.array(mesh.vertices, dtype=np.float64), np.array(mesh.faces, dtype=np.int32)
            except:
                pass
        
        # Basic OBJ loader as fallback
        if filepath.lower().endswith('.obj'):
            return self._load_obj_basic(filepath)
        
        return None, None
    
    def _load_obj_basic(self, filepath):
        """Basic OBJ file loader."""
        vertices = []
        faces = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('v '):
                        # Vertex
                        parts = line.split()
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith('f '):
                        # Face
                        parts = line.split()[1:]
                        # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
                        face_indices = []
                        for part in parts:
                            vertex_idx = int(part.split('/')[0]) - 1  # OBJ indices are 1-based
                            face_indices.append(vertex_idx)
                        
                        if len(face_indices) >= 3:
                            # Triangulate if necessary
                            for i in range(1, len(face_indices) - 1):
                                faces.append([face_indices[0], face_indices[i], face_indices[i + 1]])
            
            if vertices and faces:
                return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)
        except Exception as e:
            print(f"Error loading OBJ file {filepath}: {e}")
        
        return None, None
    
    def _update_mesh_display(self):
        """Update the displayed mesh to the current frame."""
        if not self.meshes:
            return
        
        vertices, faces = self.meshes[self.current_frame]
        
        # Register or update the mesh
        ps_mesh = ps.register_surface_mesh(self.mesh_name, vertices, faces)
        ps_mesh.set_color([0.7, 0.7, 0.9])  # Light blue color
    
    def _setup_view(self, vertices):
        """Setup optimal camera view for the mesh."""
        # Compute bounding box
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_center = (bbox_min + bbox_max) / 2.0
        bbox_size = np.max(bbox_max - bbox_min)
        
        # Set automatic scene extents computation
        ps.set_automatically_compute_scene_extents(True)
        
        # Use perspective projection for better depth perception
        ps.set_view_projection_mode("perspective")
        
        # Set up nice lighting and ground plane
        try:
            ps.set_ground_plane_mode("shadow_only")
            ps.set_ground_plane_height_factor(0.0)
        except:
            # Fallback if ground plane settings fail
            pass
        
        # Reset camera to show the mesh nicely
        ps.reset_camera_to_home_view()
    
    def _gui_callback(self):
        """ImGui callback for animation controls."""
        # Handle video export frame by frame in main thread
        psim.TextUnformatted(f"Mesh Sequence Viewer")
        psim.Separator()
        
        psim.TextUnformatted(f"Frame: {self.current_frame + 1}/{len(self.meshes)}")
        psim.TextUnformatted(f"File: {os.path.basename(self.mesh_files[self.current_frame])}")
        
        # Frame slider
        changed, new_frame = psim.SliderInt("Frame", self.current_frame, 0, len(self.meshes) - 1)
        if changed:
            self.current_frame = new_frame
            self._update_mesh_display()
        
        psim.Separator()
        
        # Play/Pause button
        if psim.Button("Play" if not self.playing else "Pause"):
            self.playing = not self.playing
        
        psim.SameLine()
        
        # Reset button
        if psim.Button("Reset"):
            self.current_frame = 0
            self.playing = False
            self._update_mesh_display()
        
        psim.SameLine()
        
        # Reset view button
        if psim.Button("Reset View"):
            if self.meshes:
                vertices, _ = self.meshes[self.current_frame]
                self._setup_view(vertices)
        
        # Frame rate slider
        psim.Separator()
        changed, new_rate = psim.SliderFloat("Frame Rate (FPS)", self.frame_rate, 1.0, 60.0)
        if changed:
            self.frame_rate = new_rate
        
        # Progress bar
        progress = (self.current_frame + 1) / len(self.meshes)
        psim.ProgressBar(progress)
        
        # Mesh info
        if self.meshes:
            vertices, faces = self.meshes[self.current_frame]
            psim.Separator()
            psim.TextUnformatted(f"Vertices: {vertices.shape[0]}")
            psim.TextUnformatted(f"Faces: {faces.shape[0]}")
        
        # Video Export Section
        if HAS_CV2:
            psim.Separator()
            psim.TextUnformatted("Video Export")
            
            # Output info
            psim.Separator()
            psim.TextUnformatted("Output Locations:")
            screenshots_path = str(self.folder_path / "screenshots")
            psim.TextUnformatted(f"Screenshots: {screenshots_path}")
            video_name = f"{os.path.basename(self.folder_path)}_sequence.mp4"
            psim.TextUnformatted(f"Video: {video_name}")
            
            # Export screenshots button
            if psim.Button("Export Video"):
                screenshots_dir = self.folder_path / "screenshots"
                screenshots_dir.mkdir(exist_ok=True)
                print(f"Exporting {len(self.meshes)} screenshots to: {screenshots_dir}")
                
                success_count = 0
                for frame_index in range(len(self.meshes)):
                    if (frame_index % 10 == 0):
                        print(f"Exporting frame {frame_index+1}/{len(self.meshes)}")
                    success = self._export_video_frame(frame_index, str(screenshots_dir))
                    if success:
                        success_count += 1
                    else:
                        print(f"FAILED: Failed to export frame {frame_index}")
                        break
                
                print(f"Screenshot export complete: {success_count}/{len(self.meshes)} frames captured")
                if success_count == len(self.meshes):
                    print("All screenshots captured successfully! Now export the video.")
                    success = self._create_video_from_frames(str(screenshots_dir), str(self.folder_path / video_name), self.frame_rate)
                    if success:
                        print("Video export complete!")
                    else:
                        print("Video export failed - see error messages above")
            
        else:
            psim.Separator()
            psim.TextUnformatted("Video Export: OpenCV required")
            psim.TextUnformatted("Install: pip install opencv-python")
    
    def _update_animation(self):
        """Update animation if playing."""
        if not self.playing:
            return
        
        current_time = time.time()
        if current_time - self.last_update_time > 1.0 / self.frame_rate:
            self.current_frame = (self.current_frame + 1) % len(self.meshes)
            self._update_mesh_display()
            self.last_update_time = current_time
    
    def _create_video_from_frames(self, frame_dir, output_path, fps):
        """Create video from a directory of frame images."""
        if not HAS_CV2:
            return False
        
        try:
            # Get list of frame files
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png')])
            if not frame_files:
                return False
            
            # Read first frame to get dimensions
            first_frame_path = os.path.join(frame_dir, frame_files[0])
            first_frame = cv2.imread(first_frame_path)
            height, width, _ = first_frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames to video
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frame_dir, frame_file)
                frame = cv2.imread(frame_path)
                out.write(frame)
                
                # Update progress
                self.export_progress = (i + 1) / len(frame_files)
            
            out.release()
            return True
            
        except Exception as e:
            print(f"Error creating video: {e}")
            return False
    
    def _export_video_frame(self, frame_index, temp_dir):
        """Export a single frame - called from main thread during GUI update."""
        if frame_index >= len(self.meshes):
            return False
        
        try:
            # Update mesh display to current export frame
            old_frame = self.current_frame
            self.current_frame = frame_index
            self._update_mesh_display()

            ps.screenshot(filename = os.path.join(temp_dir, f"frame_{frame_index:06d}.png"))
            
            # Restore original frame
            self.current_frame = old_frame
            self._update_mesh_display()
            return True

        except Exception as e:
            print(f"Error capturing frame {frame_index}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Run the mesh sequence viewer."""
        ps.init()
        ps.set_program_name("Mesh Sequence Viewer")
        
        # Configure Polyscope settings (with error handling for version compatibility)
        try:
            ps.set_up_dir("z_up")  # Use Z-up convention
            ps.set_front_dir("neg_y_front")  # Set front direction
            ps.set_navigation_style("turntable")  # Good for mesh viewing
            ps.set_background_color([1.0, 1.0, 1.0])  # White background
        except Exception as e:
            print(f"Warning: Some Polyscope settings failed: {e}")
            # Continue with defaults
        
        # Set up the initial mesh
        self._update_mesh_display()
        
        # Set up initial view (only once at startup)
        if self.meshes:
            vertices, _ = self.meshes[0]
            self._setup_view(vertices)
        
        # Set up GUI callback
        def combined_callback():
            # Update animation
            self._update_animation()
            # GUI callback
            self._gui_callback()
        
        ps.set_user_callback(combined_callback)
        
        # Show the viewer (this will block until window is closed)
        ps.show()


def main():
    parser = argparse.ArgumentParser(description='View mesh sequence animation using Polyscope')
    parser.add_argument('folder', nargs='?', default='.', 
                       help='Folder containing mesh sequence files (default: current directory)')
    parser.add_argument('--fps', type=float, default=10.0,
                       help='Initial frame rate for animation (default: 10.0)')
    parser.add_argument('--end_frame', type=int, default=None,
                       help='End frame for animation (default: None)')
    args = parser.parse_args()
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        sys.exit(1)
    
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        sys.exit(1)
    
    try:
        viewer = MeshSequenceViewer(folder_path, args.end_frame)
        viewer.frame_rate = args.fps
        if args.end_frame is not None:
            viewer.end_frame = args.end_frame
        print("\nControls:")
        print("- Use the GUI controls to play/pause and navigate frames")
        print("- Press Ctrl+C to exit")
        print("")
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
