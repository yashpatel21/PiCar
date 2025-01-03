from pycoral.utils.edgetpu import list_edge_tpus

devices = list_edge_tpus()
if devices:
    print(f"Found {len(devices)} Edge TPU device(s): {devices}")
else:
    print("No Edge TPU devices found.")