# simulate_clients.py
import requests, base64, numpy as np, json, time

SERVER = "http://127.0.0.1:8080"

def arrays_to_b64(arrays):
    flat = np.concatenate([a.astype(np.float32).ravel() for a in arrays])
    return base64.b64encode(flat.tobytes()).decode("ascii")

def upload(client_id, n_samples, arrays):
    b64 = arrays_to_b64(arrays)
    shapes = [list(a.shape) for a in arrays]
    payload = {"client_id": client_id, "n_samples": n_samples, "weights_base64": b64, "shapes": shapes}
    r = requests.post(f"{SERVER}/upload_update", json=payload)
    print("upload", client_id, r.status_code, r.text)

if __name__ == "__main__":
    shapes = [(8,), (8,8)]
    arrs1 = [np.ones(s, dtype=np.float32)*1.0 for s in shapes]
    arrs2 = [np.ones(s, dtype=np.float32)*3.0 for s in shapes]
    upload("client1", 10, arrs1)
    upload("client2", 30, arrs2)
    time.sleep(1)
    r = requests.post(f"{SERVER}/aggregate")
    print("aggregate", r.status_code, r.text)
