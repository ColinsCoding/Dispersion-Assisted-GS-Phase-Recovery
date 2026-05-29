// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title  OpticalPhaseVerifier
 * @author Jalali Lab
 *
 * Explicit trust for optical phase retrieval measurements.
 *
 * Problem
 * ───────
 * A lab measures I₁(t) and I₂(t) through two dispersive fibres (D₁=−600 ps²,
 * D₂=−1200 ps²) and runs GS phase retrieval to recover φ(t).
 * Without a trusted third party, how does anyone verify the claimed φ
 * was actually derived from the claimed measurements?
 *
 * Solution — hash commitments on-chain
 * ─────────────────────────────────────
 * Step 5  Compute C_I1 = keccak256(I1_bytes), C_I2 = keccak256(I2_bytes),
 *                 C_φ  = keccak256(phi_bytes)   off-chain (Python).
 * Step 6  submit(C_I1, C_I2)        → stores commitment, returns measurement id.
 *         attachSolution(id, C_φ)   → locks in the phase solution hash.
 *         verify(id, I1_raw, I2_raw)→ returns true iff raw data matches stored hashes.
 *
 * Explicit trust: no party is trusted. The EVM enforces:
 *   - Only the original submitter can attach a solution.
 *   - Once attached, the commitment is immutable.
 *   - Anyone with the raw data can verify provenance.
 *   - A 1-bit tamper in I1 produces a different keccak256 → verify() returns false.
 *
 * Odd / even connection
 * ──────────────────────
 * The FFT used in GS phase retrieval is a Cooley-Tukey radix-2 circuit:
 *   X[k]       = FFT_even[k] + ω^k · FFT_odd[k]    (butterfly gate, even out)
 *   X[k + N/2] = FFT_even[k] - ω^k · FFT_odd[k]    (butterfly gate, odd  out)
 * This is arithmetically equivalent to an R1CS circuit over F_p (BN128 prime).
 * The same keccak256 hash used here is also the hash function in Ethereum's
 * Patricia-Merkle trie, ZK-SNARK public input hashing, and EIP-712 signing.
 *
 * Deploy (local Hardhat / Foundry):
 *   forge create ethereum/OpticalPhaseVerifier.sol:OpticalPhaseVerifier \
 *         --rpc-url http://localhost:8545 --private-key <key>
 *
 * Deploy (Polygon Amoy testnet, free):
 *   forge create ... --rpc-url https://rpc-amoy.polygon.technology
 *
 * Interact from Python (web3.py):
 *   see  ethereum/interact.py
 */
contract OpticalPhaseVerifier {

    // ── Data structures ───────────────────────────────────────────────────────

    struct Measurement {
        bytes32 I1_hash;    // keccak256 of D1 intensity array (float32 LE bytes)
        bytes32 I2_hash;    // keccak256 of D2 intensity array
        bytes32 phi_hash;   // keccak256 of GS phase solution (0 until attached)
        uint256 timestamp;  // block.timestamp of submission
        address submitter;  // address that called submit()
        bool    solved;     // true once attachSolution() is called
    }

    // ── State ─────────────────────────────────────────────────────────────────

    mapping(uint256 => Measurement) public measurements;
    uint256 public measurementCount;

    // ── Events ────────────────────────────────────────────────────────────────

    event MeasurementSubmitted(
        uint256 indexed id,
        address indexed submitter,
        bytes32 I1_hash,
        bytes32 I2_hash
    );
    event SolutionAttached(
        uint256 indexed id,
        bytes32 phi_hash,
        uint256 timestamp
    );

    // ── Step 6a: Submit measurement commitments ───────────────────────────────

    /**
     * @notice Submit keccak256 commitments of I1 and I2.
     * @param  I1_hash  keccak256(abi.encodePacked(I1_float32_bytes))
     * @param  I2_hash  keccak256(abi.encodePacked(I2_float32_bytes))
     * @return id       Measurement index (use in attachSolution / verify)
     */
    function submit(bytes32 I1_hash, bytes32 I2_hash)
        external returns (uint256 id)
    {
        id = measurementCount++;
        measurements[id] = Measurement({
            I1_hash:   I1_hash,
            I2_hash:   I2_hash,
            phi_hash:  bytes32(0),
            timestamp: block.timestamp,
            submitter: msg.sender,
            solved:    false
        });
        emit MeasurementSubmitted(id, msg.sender, I1_hash, I2_hash);
    }

    // ── Step 6b: Attach phase solution ────────────────────────────────────────

    /**
     * @notice Lock in the GS phase solution for measurement `id`.
     *         Only the original submitter may call this, and only once.
     * @param  id        Measurement id returned by submit()
     * @param  phi_hash  keccak256(abi.encodePacked(phi_float32_bytes))
     */
    function attachSolution(uint256 id, bytes32 phi_hash) external {
        require(id < measurementCount,             "unknown id");
        require(msg.sender == measurements[id].submitter, "not submitter");
        require(!measurements[id].solved,          "solution already attached");
        measurements[id].phi_hash = phi_hash;
        measurements[id].solved   = true;
        emit SolutionAttached(id, phi_hash, block.timestamp);
    }

    // ── Step 6c: Verify raw data ──────────────────────────────────────────────

    /**
     * @notice Verify that raw byte arrays match the stored commitments.
     *         Anyone can call this with the raw measurement data to confirm
     *         provenance — no trusted intermediary required.
     *
     * @param  id      Measurement id
     * @param  I1_raw  Raw float32 LE bytes of I1 (same bytes hashed off-chain)
     * @param  I2_raw  Raw float32 LE bytes of I2
     * @return bool    true  → data matches commitment exactly
     *                 false → data was tampered or wrong id
     */
    function verify(
        uint256 id,
        bytes calldata I1_raw,
        bytes calldata I2_raw
    ) external view returns (bool) {
        require(id < measurementCount, "unknown id");
        return keccak256(I1_raw) == measurements[id].I1_hash
            && keccak256(I2_raw) == measurements[id].I2_hash;
    }

    /**
     * @notice Verify a claimed phase solution against the stored hash.
     */
    function verifySolution(uint256 id, bytes calldata phi_raw)
        external view returns (bool)
    {
        require(id < measurementCount, "unknown id");
        require(measurements[id].solved, "no solution attached");
        return keccak256(phi_raw) == measurements[id].phi_hash;
    }

    // ── View helpers ──────────────────────────────────────────────────────────

    function getMeasurement(uint256 id)
        external view
        returns (bytes32 I1_hash, bytes32 I2_hash, bytes32 phi_hash,
                 uint256 ts, address submitter, bool solved)
    {
        Measurement storage m = measurements[id];
        return (m.I1_hash, m.I2_hash, m.phi_hash,
                m.timestamp, m.submitter, m.solved);
    }
}
