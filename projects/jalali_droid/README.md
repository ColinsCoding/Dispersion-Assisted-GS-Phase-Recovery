# Jalali droid: a solarpunk 3D battle in C++17/CUDA

**ジャラリ・ガーディアン (Jalali Guardian)** vs. **剣士ソウマ (Souma, the Foam Swordsman)** on a rolling-hill field with a **太陽光パネル solar farm**, in late-afternoon sun. They're original characters inspired by Japanese adventure and monster-battle games, not copies of any franchise's.

The Guardian is a pan-tilt-zoom camera droid themed on Bahram Jalali's lab. It **ray-traces** its own camera view (GPU or CPU, same code), finds the swordsman with the **Phase Stretch Transform** (PST, from Jalali's PhyCV), aims with **motor differential equations** plus velocity feed-forward, steadies its camera on an **isolation mount**, and fires a **3D game laser**. The laser can hit the body, get **parried by the foam sword**, or be **blocked by a hill**. Damage follows a monster-battle formula with types. Each frame also computes the solar farm's **photovoltaic output**, including the shadows that fall on it. Everything is defined in `assets/characters.def`. The laser is simulated and eye-safe; nothing here drives hardware.

| Subject | What the code does | File |
|---|---|---|
| 3D world | heightfield terrain `z = h(x, y)` with analytic normal; body/head spheres; foam-sword capsule; hops `z = v₀t − gt²/2` | `include/scene3d.hpp`, `src/main3d.cpp` |
| Solar farm | 3 × 7 PV panels (rectangles) on posts (capsules), tilted at the site latitude (38.6°, Sacramento), facing the sun's azimuth | `include/scene3d.hpp`, `src/defs.cpp` |
| Lighting (solarpunk) | warm sun with **shadow rays**; near-neutral sky light + green ground bounce; Blinn-Phong glint on panel glass and posts; sky with zenith→horizon gradient, sun disk and halo; distance haze | `include/scene3d.hpp` |
| Rendering | one ray per pixel: analytic sphere / capsule / rectangle hits, ray-march + bisection for the terrain. The `HD` functions compile for **both** GPU and CPU | `include/scene3d.hpp`, `src/render3d.cu`, `src/defs.cpp` |
| Photovoltaics | `P = η·A·(1 − dirt/2)·(DNI·cos θ·f_unshaded + DHI·(1+n_z)/2)` per panel (fully dusty glass passes half the light); the unshaded fraction comes from shadow rays at 3×3 points per panel | `src/defs.cpp`, `src/main3d.cpp` |
| Computer vision | orange **chromaticity** feature (survives shading) on a 0.2 pedestal → PST → centroid of the positive phase | `src/vision_*.cpp/.cu` |
| Mechanics + E&M | per-axis DC gimbal motor: `J ω' = K_t i − b ω`, `L di/dt = V − R i − K_e ω`, RK4 at 0.1 ms; velocity feed-forward | `src/physics.cpp`, `src/main3d.cpp` |
| Isolation table | torsional head mount against the droid's 12 Hz gait | `src/physics.cpp` |
| Photonics | Gaussian beam `w(z)`; in 3D, the laser is a traced ray | `src/physics.cpp`, `scene3d.hpp` |
| Battle | `⌊⌊⌊2L/5+2⌋·P·A/D⌋/50⌋+2` × same-type bonus 1.5 × type chart × roll 0.85–1.00 | `src/defs.cpp`, `assets/characters.def` |
| Definitions | UTF-8 INI: Japanese + English names, stats, moves, type chart, world, sun, solar farm | `assets/characters.def` |
| NumPy | an **independent** float64 reference renderer (now with panels, posts, shadows, sky, haze) that cross-checks C++ frames; 3D arena GIF; camera-view GIF with a PV readout | `python/droid3d_np.py` |
| CLI | options, CSV log (now with `pv_kw`, `shaded_panels`, `droid_motor_w`), PPM frames, `$?` exit codes | `src/main3d.cpp`, `src/main.cpp` (2D) |

## Build, test, play

```bash
./build.sh            # Linux / WSL: builds 2D + 3D, runs 49 tests, plays the 3D battle
```
```bat
build.bat             :: Windows (MSVC + nvcc + cuFFT)
```
Then render the visuals:
```bash
out/droid3d --frames 1800 --frames-dir out/frames --frame-every 3 --dump-frame 290 --dump out/frame.bin
python python/droid3d_np.py check  out/frame.bin                          # NumPy re-renders, compares
python python/droid3d_np.py arena  out/droid3d_run.csv out/arena3d.gif     # 3D world + solar farm, orbiting view
python python/droid3d_np.py camera out/frames out/droid3d_run.csv out/camera.gif   # the droid's eye, with PV readout
```
Edit stats, moves, the type chart, the sun direction or the solar farm in `assets/characters.def`; no rebuild needed.

`droid3d` exit codes: `0` ok, `2` bad arguments, `3` GPU requested but unavailable, `4` cannot write output, `5` definitions file missing or invalid.

## Play it (Windows): `out\droid_play.exe`

You are 剣士ソウマ. The Guardian hunts you with the same PST vision and motor physics as above, on its own ray-traced camera.

1. **1人称・農場 (first person, farming).** Mouse look (the cursor is captured), WASD to walk and strafe, Space to jump. The panels start 25–75% dusty and keep collecting dust. Stand at one and hold **E** to clean it. The HUD shows the farm's real output against the clean ideal (5.96 kW), and that ratio fills your **solar charge** (about 45 s at a clean farm). The droid's laser only reaches 9 m (`laser_range_m`), so the farm's far side is a safe zone (射程外).
2. **3人称・クランチタイム (third person, crunch time).** When either fighter drops to 30% HP, the camera cuts to a chase view. **Left click** = Foam Slash (within 1.8 m); **right click (hold)** = raise the sword to parry the beam.
3. **最後の賭け (the Last Gamble).** In crunch time, with full charge and your HP at or below 30%, press **Q** for a 50/50 roll. Win: the Guardian freezes for 3 s and your next slash does ×2. Lose: 10 HP of recoil.

**V** switches camera at any time; **Esc** quits. Sound effects (laser, slash, parry, hit, gamble) are synthesized in code as WAVs and played with `PlaySound`, so there are no asset files. Options: `--cpu`, `--mute`, `--autoplay N --screenshot PREFIX`. Autoplay is a scripted bot that farms, soaks hits until crunch time, then gambles and slashes; it saves `PREFIX_1st.bmp`, `PREFIX_3rd.bmp` and `PREFIX_end.bmp`. Those images hold the 3D view, HP/charge bars and crosshair; the HUD text is drawn onto the window, so it is not in them.

Bot run, 3600 frames (2026-10-26): dust fell from 53% to 19% and the farm rose from 3.8 to 5.4 kW. The bot took 4 beams (down to 29/110 HP) and won its gamble, and the ×2 slash finished the Guardian in 2 slashes instead of 3. It ran at 291 fps on the GPU (game speed is capped at 30 fps when you play). Linux has no `droid_play` (Win32 window, mouse capture and sound); use `droid3d` there.

## Measured on the RTX 4060 (2026-10-26)

| | Result |
|---|---|
| Tests | Windows (CUDA 13.0, MSVC): 2D 16/16 + 3D 33/33. WSL Ubuntu (CUDA 12.0, gcc 13.3): 16/16 + 32/32, run before the soiling check was added |
| Battle, Windows | **Guardian wins with 4/95 HP** after 20.5 s: 6 beam hits, 7 parried by the foam sword, 2 Foam Slashes (×2 vs Steel); target seen 616/616 frames, mean aim error 1.25° |
| Battle, WSL | **Souma wins with 29/110 HP** after 25.2 s: 4 beam hits, 7 parries, 3 Foam Slashes; target seen 756/756 frames, mean aim error 1.29° (see below) |
| Solar farm | 21 panels, 21% modules, sun 28° up: **5.96 kW** mean (34 Wh over the 20.5 s battle) against **6.94 kW** with no shading at all |
| Shading losses | **row-to-row self-shading costs 14.2%** at this sun angle (rows 2.6 m apart); the hero's shadow dimmed the farm in 1/616 frames (Windows), 9/756 (WSL) |
| Droid power | gimbal motors alone average ~0.02 W; electronics, camera and GPU aren't modelled |
| C++/CUDA vs NumPy render | frames showing ~800 panel pixels, posts and shadows: median difference 1.8e-7–6e-7; 0.02–0.04% of pixels differ by > 0.02 (grid-line and shadow edges, float32 vs float64) |
| GPU render, 128×128 | 0.3–0.4 ms/frame (Windows), 0.65 ms (WSL), with shadow rays; the buffer is kept between frames |

### Why Windows and WSL now play different battles
Before the solarpunk upgrade, both platforms played the **identical** battle. Now they don't. The new shading (shadow rays, `powf` glints, haze, chromaticity) is evaluated in float32 by two different compilers and CUDA versions (MSVC + CUDA 13.0 vs gcc + CUDA 12.0), and they round a few pixel values differently.

A shifted pixel near the tunic's edge moves the PST centroid by a fraction of a pixel. That changes the motor reference, so the droid's aim drifts slightly (1.25° vs 1.29° mean error), so the laser lands on different frames, so different damage rolls are drawn from the shared random generator. The divergence grows until it changes who wins.

It's sensitivity to tiny differences (a chaotic game loop), not a bug: each platform reproduces its own battle exactly on every run. Bit-identical play across platforms would need fixed-point or strictly specified floating-point math in the renderer.

## Bugs found while building (kept here because each was a real lesson)
- **The foam sword biased the 2D locator** (1–2°): fixed with a tighter PST low-pass (σ 0.15 → 0.05) and the positive phase only.
- **Zero-background PST:** in 3D the feature image was exactly 0 off the target, and the *phase* of near-zero values is noise, which scattered fake edges everywhere. The droid saw the hero in only 112/756 frames and lost. A 0.2 pedestal fixed it.
- **Shading hid the tunic:** a raw "R − (G+B)/2" feature vanished on the shaded side; chromaticity (divided by brightness) doesn't.
- **Capsule end cap:** a ray along the sword's axis divided by zero in the cylinder formula; the capsule is now cylinder ∪ two spheres.
- **Solarpunk sky light blinded the droid:** a strongly blue ambient plus 45 m haze desaturated the tunic's shaded side below the vision threshold (caught by a test). Fixed with a near-neutral sky light and 120 m haze.
- **A noon sun can't be shaded by the hero:** at 60° elevation a shadow ray from a panel rises 1.7 m per metre, so the hero would have to stand inside the panel. The field moved to the anti-sun side, past the edge of his path, and the sun dropped to 28° (late afternoon).
- **Mis-attributed shading:** the first summary blamed "the hero's shadow" for shading in every frame. It was really the rows shading each other. A no-hero baseline now separates the 14.2% row-to-row loss from the hero's occasional shadow.

## Limitations
- The world is procedural, with no texture maps or animation rig. The characters are spheres and a capsule.
- Shadows are hard-edged (one shadow ray per point; no soft penumbra).
- The PV model is a clear-sky, first-order estimate: no cell-temperature derating, no bypass-diode or string behaviour when a panel is part-shaded, and the sun doesn't move during a battle.
- The battle is a simple real-time rule set (hold-to-hit, cooldowns), not a full game.
- The vision tracks a known color; a real camera would need lighting-robust detection.
