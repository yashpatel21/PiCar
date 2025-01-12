#!/usr/bin/env python3

def test_picarx_installation():
    print("Testing PiCar-X Installation")
    print("-" * 50)
    
    # First, try importing the module
    print("Step 1: Importing picarx module...")
    try:
        import picarx
        print("✓ Successfully imported picarx module")
    except ImportError as e:
        print(f"✗ Failed to import picarx: {str(e)}")
        return False
        
    # Try creating a PiCar-X instance
    print("\nStep 2: Initializing PiCar-X...")
    try:
        px = picarx.Picarx()
        print("✓ Successfully initialized PiCar-X")
    except Exception as e:
        print(f"✗ Failed to initialize PiCar-X: {str(e)}")
        return False
    
    # Try accessing some basic components
    print("\nStep 3: Testing basic components...")
    try:
        # Set steering angle to center (0 degrees)
        print("Testing steering servo...")
        px.set_dir_servo_angle(0)
        
        # Test camera servo
        print("Testing camera pan servo...")
        px.set_cam_pan_angle(0)

        # Test camera servo
        print("Testing camera tilt servo...")
        px.set_cam_tilt_angle(0)
        
        # Test motors (very slight movement)
        print("Testing motors (brief movement)...")
        px.forward(30)
        import time
        time.sleep(1)  # Move for just a fraction of a second
        px.stop()
        
        print("✓ Basic component tests completed")
    except Exception as e:
        print(f"✗ Component test failed: {str(e)}")
        return False
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    try:
        test_picarx_installation()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")