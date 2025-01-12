import time

def test_i2c():
    """
    Test I2C functionality by scanning for available devices.
    This will use the SMBus interface to scan all possible I2C addresses.
    """
    print("\nTesting I2C Interface:")
    try:
        import smbus2
        bus = smbus2.SMBus(1)  # Use I2C bus 1 (pins 3 & 5 on RPi)
        
        print("Scanning I2C bus for devices...")
        devices_found = []
        
        for address in range(0x03, 0x78):
            try:
                bus.read_byte(address)
                devices_found.append(address)
            except:
                pass
                
        if devices_found:
            print(f"Found {len(devices_found)} I2C devices at addresses:")
            for addr in devices_found:
                print(f"  0x{addr:02X}")
        else:
            print("No I2C devices found, but interface is accessible")
            
        print("✓ I2C interface test successful")
        return True
        
    except Exception as e:
        print(f"✗ I2C test failed: {str(e)}")
        return False

def test_spi():
    """
    Test SPI functionality by attempting to open and configure the SPI device.
    This will verify if we can access the SPI interface without actually sending data.
    """
    print("\nTesting SPI Interface:")
    try:
        import spidev
        spi = spidev.SpiDev()
        spi.open(0, 0)  # Open SPI port 0, device 0
        
        # Configure SPI
        spi.max_speed_hz = 500000
        spi.mode = 0
        
        print("Successfully opened SPI device")
        print("Current configuration:")
        print(f"  Max speed: {spi.max_speed_hz} Hz")
        print(f"  Mode: {spi.mode}")
        
        spi.close()
        print("✓ SPI interface test successful")
        return True
        
    except Exception as e:
        print(f"✗ SPI test failed: {str(e)}")
        return False

def main():
    print("Starting interface tests...")
    
    i2c_result = test_i2c()
    spi_result = test_spi()
    
    print("\nTest Summary:")
    print(f"I2C Test: {'Passed' if i2c_result else 'Failed'}")
    print(f"SPI Test: {'Passed' if spi_result else 'Failed'}")

if __name__ == "__main__":
    main()