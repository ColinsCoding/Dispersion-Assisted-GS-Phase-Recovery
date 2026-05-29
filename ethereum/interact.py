"""
ethereum/interact.py
====================
Python ↔ OpticalPhaseVerifier.sol via web3.py

Install:  pip install web3
Run:      python ethereum/interact.py

Requires a running EVM node.  Quickest local option:
  npx hardhat node          (needs Node.js + npm install --save-dev hardhat)
  OR
  anvil                     (Foundry: curl -L https://foundry.paradigm.xyz | bash)
"""

import sys, json, pathlib, numpy as np

try:
    from web3 import Web3
except ImportError:
    sys.exit("web3 not installed — run:  pip install web3")

# ── Connection ─────────────────────────────────────────────────────────────────
RPC_URL      = "http://127.0.0.1:8545"          # local Hardhat / Anvil
PRIVATE_KEY  = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
# ^^^ Hardhat account #0 — never use a real key here

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    sys.exit(f"[!] Cannot connect to {RPC_URL}\n    Start Hardhat: npx hardhat node")

account = w3.eth.account.from_key(PRIVATE_KEY)
print(f"[ok] Connected to {RPC_URL}  account={account.address}")

# ── Deploy contract ─────────────────────────────────────────────────────────────
# Compile with:  solc --abi --bin ethereum/OpticalPhaseVerifier.sol -o ethereum/build/
BUILD_DIR = pathlib.Path(__file__).parent / "build"
ABI_PATH  = BUILD_DIR / "OpticalPhaseVerifier.abi"
BIN_PATH  = BUILD_DIR / "OpticalPhaseVerifier.bin"

if not ABI_PATH.exists():
    sys.exit(f"[!] ABI not found at {ABI_PATH}\n"
             "    Compile first:\n"
             "      solc --abi --bin ethereum/OpticalPhaseVerifier.sol -o ethereum/build/\n"
             "    Or use Foundry:\n"
             "      forge build --root ethereum/")

abi = json.loads(ABI_PATH.read_text())
bytecode = "0x" + BIN_PATH.read_text().strip()

Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash  = Contract.constructor().transact({"from": account.address})
receipt  = w3.eth.wait_for_transaction_receipt(tx_hash)
contract = w3.eth.contract(address=receipt.contractAddress, abi=abi)
print(f"[ok] Deployed at {receipt.contractAddress}")


# ── Helper: keccak256 of float32 array ─────────────────────────────────────────
def keccak_f32(arr: np.ndarray) -> bytes:
    raw = arr.astype(np.float32).tobytes()
    return Web3.keccak(raw)          # Ethereum-native keccak256


# ── Simulate optical measurements ──────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "optical_dashboard"))
import dsp as DSP

result = DSP.optical_zk_demo(N=64)
I1     = np.array(result["I1"])
I2     = np.array(result["I2"])
phi    = np.array(result["phi_est"])

C_I1 = keccak_f32(I1)
C_I2 = keccak_f32(I2)
C_phi = keccak_f32(phi)
print(f"\n[step 5] Commitments")
print(f"  C_I1 = 0x{C_I1.hex()}")
print(f"  C_I2 = 0x{C_I2.hex()}")
print(f"  C_φ  = 0x{C_phi.hex()}")


# ── Step 6a: submit() ──────────────────────────────────────────────────────────
tx   = contract.functions.submit(C_I1, C_I2).transact({"from": account.address})
rcpt = w3.eth.wait_for_transaction_receipt(tx)
meas_id = contract.functions.measurementCount().call() - 1
print(f"\n[step 6a] submit() — measurement id={meas_id}  gas={rcpt.gasUsed}")


# ── Step 6b: attachSolution() ──────────────────────────────────────────────────
tx   = contract.functions.attachSolution(meas_id, C_phi).transact({"from": account.address})
rcpt = w3.eth.wait_for_transaction_receipt(tx)
print(f"[step 6b] attachSolution() — gas={rcpt.gasUsed}")


# ── Step 6c: verify() ──────────────────────────────────────────────────────────
I1_raw = I1.astype(np.float32).tobytes()
I2_raw = I2.astype(np.float32).tobytes()

ok = contract.functions.verify(meas_id, I1_raw, I2_raw).call()
print(f"\n[step 6c] verify(id, I1_raw, I2_raw) = {ok}")      # True

# Tamper test: flip one bit in I1
tampered       = I1.copy(); tampered[0] += 1.0
tampered_raw   = tampered.astype(np.float32).tobytes()
ok_tamper      = contract.functions.verify(meas_id, tampered_raw, I2_raw).call()
print(f"          verify(tampered I1)          = {ok_tamper}")  # False

phi_raw = phi.astype(np.float32).tobytes()
ok_phi  = contract.functions.verifySolution(meas_id, phi_raw).call()
print(f"          verifySolution(phi)          = {ok_phi}")     # True

print("\n[done] Explicit trust established — no trusted third party used.")
