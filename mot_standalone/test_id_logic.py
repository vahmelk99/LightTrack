"""
Test ID logic - verify IDs are tracked correctly and removal works by ID not position
"""

import numpy as np
import sys

print("Testing ID Logic...")
print("=" * 60)

# Import
try:
    from mot_tracker import MultiObjectTracker
    from easydict import EasyDict as edict
    print("✓ Imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create dummy components
info = edict()
info.stride = 16
info.dataset = 'VOT2019'

args = edict()
args.even = 0

# Create dummy frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Initialize tracker (without model for testing)
print("Test 1: ID Assignment and Tracking")
print("-" * 60)

# We'll mock the tracker to avoid needing the actual model
class MockTracker:
    def __init__(self, info=None, even=0, device='cpu'):
        # Accept all parameters but don't use them
        pass
    
    def init(self, frame, target_pos, target_sz, model):
        return {
            'target_pos': target_pos,
            'target_sz': target_sz,
            'p': None,
            'net': None,
            'avg_chans': None,
            'window': None
        }
    
    def track(self, state, frame):
        # Just return the state unchanged for testing
        return state

# Monkey patch for testing
import mot_tracker
original_lighttrack = mot_tracker.Lighttrack
mot_tracker.Lighttrack = MockTracker

try:
    tracker = MultiObjectTracker(info, None, args, device='cpu')
    
    # Add 4 objects
    print("\n1. Adding 4 objects:")
    box1 = [10, 10, 50, 50]
    box2 = [60, 60, 100, 100]
    box3 = [110, 110, 150, 150]
    box4 = [160, 160, 200, 200]
    
    id0 = tracker.add_object(frame, box1)
    id1 = tracker.add_object(frame, box2)
    id2 = tracker.add_object(frame, box3)
    id3 = tracker.add_object(frame, box4)
    
    print(f"   Added objects with IDs: {id0}, {id1}, {id2}, {id3}")
    assert id0 == 0 and id1 == 1 and id2 == 2 and id3 == 3, "IDs should be 0, 1, 2, 3"
    print("   ✓ IDs assigned correctly: 0, 1, 2, 3")
    
    # Track and verify IDs
    print("\n2. Tracking objects and verifying IDs:")
    results = tracker.track_all(frame)
    tracked_ids = [obj_id for obj_id, _ in results]
    print(f"   Tracked IDs: {tracked_ids}")
    assert tracked_ids == [0, 1, 2, 3], f"Expected [0, 1, 2, 3], got {tracked_ids}"
    print("   ✓ All objects tracked with correct IDs")
    
    # Remove object with ID 1 (not position 1, but actual ID 1)
    print("\n3. Removing object with ID 1:")
    success = tracker.remove_object(1)
    assert success, "Remove should succeed"
    print("   ✓ Object with ID 1 removed")
    
    # Track again and verify IDs
    print("\n4. Tracking after removal:")
    results = tracker.track_all(frame)
    tracked_ids = [obj_id for obj_id, _ in results]
    print(f"   Tracked IDs: {tracked_ids}")
    assert tracked_ids == [0, 2, 3], f"Expected [0, 2, 3], got {tracked_ids}"
    print("   ✓ Remaining objects have correct IDs (no shifting!)")
    
    # Verify ID 1 is available for reuse
    print("\n5. Checking available IDs:")
    available = tracker.available_ids
    print(f"   Available IDs: {available}")
    assert 1 in available, "ID 1 should be available for reuse"
    print("   ✓ ID 1 is available for reuse")
    
    # Add new object - should get ID 1
    print("\n6. Adding new object (should reuse ID 1):")
    box5 = [210, 210, 250, 250]
    new_id = tracker.add_object(frame, box5)
    print(f"   New object got ID: {new_id}")
    assert new_id == 1, f"Expected ID 1 (reused), got {new_id}"
    print("   ✓ ID 1 reused correctly")
    
    # Track and verify all IDs
    print("\n7. Final tracking verification:")
    results = tracker.track_all(frame)
    tracked_ids = [obj_id for obj_id, _ in results]
    print(f"   Tracked IDs: {tracked_ids}")
    assert tracked_ids == [0, 2, 3, 1], f"Expected [0, 2, 3, 1], got {tracked_ids}"
    print("   ✓ All objects tracked with correct IDs")
    
    # Remove object with ID 2 (which is at position 1 in array)
    print("\n8. Removing object with ID 2 (at array position 1):")
    success = tracker.remove_object(2)
    assert success, "Remove should succeed"
    print("   ✓ Object with ID 2 removed")
    
    # Track and verify
    print("\n9. Tracking after second removal:")
    results = tracker.track_all(frame)
    tracked_ids = [obj_id for obj_id, _ in results]
    print(f"   Tracked IDs: {tracked_ids}")
    assert tracked_ids == [0, 3, 1], f"Expected [0, 3, 1], got {tracked_ids}"
    print("   ✓ Correct objects remain with correct IDs")
    
    # Try to remove non-existent ID
    print("\n10. Trying to remove non-existent ID 99:")
    success = tracker.remove_object(99)
    assert not success, "Remove should fail for non-existent ID"
    print("   ✓ Correctly returns False for non-existent ID")
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nKey Points Verified:")
    print("  ✓ IDs are tracked in state, not by array position")
    print("  ✓ remove_object() removes by ID, not array position")
    print("  ✓ IDs don't shift when objects are removed")
    print("  ✓ Removed IDs are available for reuse")
    print("  ✓ Returns False for non-existent IDs")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Restore original
    mot_tracker.Lighttrack = original_lighttrack
