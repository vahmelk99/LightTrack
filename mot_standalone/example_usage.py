"""
Example usage of the MOT (Multi-Object Tracker) wrapper

This demonstrates how to use the MOT class for tracking multiple objects.
"""

from mot_wrapper import MOT
import cv2
import numpy as np


device = 'cpu' # 'cuda'

def example_with_camera():
    """Example: Track objects from camera feed"""
    
    # Initialize tracker
    print("Initializing tracker...")
    mot = MOT(
        model_path='../snapshot/LightTrackM/LightTrackM.pth',
        device=device  # or 'cpu'
    )
    print("Tracker ready!")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera")
        return
    
    # Add some initial objects (example boxes)
    # In practice, you would get these from user selection or detection
    print("\nAdding initial objects...")
    obj_id_1 = mot.add_box(frame, [100, 100, 200, 200])  # [x1, y1, x2, y2]
    print(f"  Added object with ID: {obj_id_1}")
    
    obj_id_2 = mot.add_box(frame, [300, 300, 400, 400])
    print(f"  Added object with ID: {obj_id_2}")
    
    print(f"\nCurrently tracking {mot.get_num_objects()} objects")
    
    # Tracking loop
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track all objects
        results = mot.track(frame)
        
        # Process results
        for obj_id, box in results:
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID: {obj_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('MOT Example', frame)
        
        # Example: Remove object after 100 frames
        if frame_count == 100:
            print(f"\nRemoving object with ID: {obj_id_1}")
            mot.remove_box(obj_id_1)
            print(f"Now tracking {mot.get_num_objects()} objects")
        
        # Example: Add new object after 150 frames
        if frame_count == 150:
            print(f"\nAdding new object...")
            new_id = mot.add_box(frame, [500, 200, 600, 300])
            print(f"  Added object with ID: {new_id} (reused ID {obj_id_1})")
            print(f"Now tracking {mot.get_num_objects()} objects")
        
        frame_count += 1
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_id_reuse():
    """Example: Demonstrate ID reuse behavior"""
    
    print("\n" + "="*50)
    print("ID REUSE DEMONSTRATION")
    print("="*50)
    
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize tracker
    mot = MOT(
        model_path='../snapshot/LightTrackM/LightTrackM.pth',
        device=device
    )
    
    # Add 4 objects
    print("\n1. Adding 4 objects:")
    id0 = mot.add_box(frame, [10, 10, 50, 50])
    id1 = mot.add_box(frame, [60, 60, 100, 100])
    id2 = mot.add_box(frame, [110, 110, 150, 150])
    id3 = mot.add_box(frame, [160, 160, 200, 200])
    print(f"   IDs: {id0}, {id1}, {id2}, {id3}")
    print(f"   Total objects: {mot.get_num_objects()}")
    
    # Remove object with ID 1
    print("\n2. Removing object with ID 1:")
    mot.remove_box(1)
    print(f"   Remaining IDs: 0, 2, 3")
    print(f"   Total objects: {mot.get_num_objects()}")
    print(f"   Available IDs for reuse: {mot.get_available_ids()}")
    
    # Remove object with ID 2
    print("\n3. Removing object with ID 2:")
    mot.remove_box(2)
    print(f"   Remaining IDs: 0, 3")
    print(f"   Total objects: {mot.get_num_objects()}")
    print(f"   Available IDs for reuse: {mot.get_available_ids()}")
    
    # Add new object - should get ID 1 (lowest available)
    print("\n4. Adding new object:")
    new_id = mot.add_box(frame, [210, 210, 250, 250])
    print(f"   New object got ID: {new_id} (reused)")
    print(f"   Total objects: {mot.get_num_objects()}")
    print(f"   Available IDs for reuse: {mot.get_available_ids()}")
    
    # Add another object - should get ID 2 (next lowest available)
    print("\n5. Adding another object:")
    new_id2 = mot.add_box(frame, [260, 260, 300, 300])
    print(f"   New object got ID: {new_id2} (reused)")
    print(f"   Total objects: {mot.get_num_objects()}")
    print(f"   Available IDs for reuse: {mot.get_available_ids()}")
    
    # Add one more - should get ID 4 (next in sequence)
    print("\n6. Adding one more object:")
    new_id3 = mot.add_box(frame, [310, 310, 350, 350])
    print(f"   New object got ID: {new_id3} (new ID)")
    print(f"   Total objects: {mot.get_num_objects()}")
    print(f"   Available IDs for reuse: {mot.get_available_ids()}")
    
    print("\n" + "="*50)
    print("SUMMARY: IDs are reused in order (lowest first)")
    print("Current IDs: 0, 1, 2, 3, 4")
    print("="*50 + "\n")


def example_basic_api():
    """Example: Basic API usage without camera"""
    
    print("\n" + "="*50)
    print("BASIC API USAGE")
    print("="*50)
    
    # Create dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Initialize tracker
    print("\nInitializing MOT...")
    mot = MOT(
        model_path='../snapshot/LightTrackM/LightTrackM.pth',
        arch='LightTrackM_Subnet',
        stride=16,
        device=device
    )
    print("✓ MOT initialized")
    
    # Add boxes
    print("\nAdding objects...")
    box1 = [100, 100, 200, 200]  # [x1, y1, x2, y2]
    box2 = [300, 300, 400, 400]
    
    id1 = mot.add_box(frame, box1)
    id2 = mot.add_box(frame, box2)
    print(f"✓ Added object {id1} at {box1}")
    print(f"✓ Added object {id2} at {box2}")
    
    # Track
    print("\nTracking...")
    results = mot.track(frame)
    print(f"✓ Tracked {len(results)} objects:")
    for obj_id, box in results:
        print(f"  - ID {obj_id}: {[int(v) for v in box]}")
    
    # Remove
    print(f"\nRemoving object {id1}...")
    success = mot.remove_box(id1)
    print(f"✓ Removed: {success}")
    print(f"  Objects remaining: {mot.get_num_objects()}")
    
    # Clear all
    print("\nClearing all objects...")
    mot.clear_all()
    print(f"✓ All cleared")
    print(f"  Objects remaining: {mot.get_num_objects()}")
    
    print("\n" + "="*50 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MOT Usage Examples')
    parser.add_argument('--example', type=str, default='api',
                       choices=['camera', 'id_reuse', 'api'],
                       help='Which example to run')
    args = parser.parse_args()
    
    if args.example == 'camera':
        example_with_camera()
    elif args.example == 'id_reuse':
        example_id_reuse()
    elif args.example == 'api':
        example_basic_api()
