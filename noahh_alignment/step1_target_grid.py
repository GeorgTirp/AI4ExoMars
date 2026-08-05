"""Step 1 -- read the target grid + legends from the label files and record them.

Writes derived/target_grid.json and prints the legend class tables. These grid
parameters (CRS, transform, size, bounds, pixel size) are the reference every
later step warps/decodes onto.
"""

from __future__ import annotations

import json

from common import (
    DC_LABEL,
    DC_LEGEND,
    DERIVED_DIR,
    IG_LEGEND,
    parse_legend,
    read_target_grid,
)


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    grid = read_target_grid(DC_LABEL)

    out = DERIVED_DIR / "target_grid.json"
    out.write_text(json.dumps(grid.to_dict(), indent=2))

    print("=== TARGET GRID (from DC label mosaic) ===")
    print(f"  size (WxH)  : {grid.width} x {grid.height}")
    print(f"  pixel size  : {grid.res_x:.9f} x {grid.res_y:.9f} m")
    print(f"  bounds      : L={grid.left:.4f} B={grid.bottom:.4f} "
          f"R={grid.right:.4f} T={grid.top:.4f}")
    print(f"  origin (TL) : x={grid.left:.4f} y={grid.top:.4f}")
    print(f"  proj4       : {grid.crs_proj4}")
    print(f"  -> wrote {out}")

    for name, path in (("DC (Descriptive Classes)", DC_LEGEND),
                       ("IG (Interpretive Groups)", IG_LEGEND)):
        entries = parse_legend(path)
        print(f"\n=== {name}: {len(entries)} classes ===")
        for e in entries:
            print(f"  {e.class_id:2d}  {e.hex}  {e.name}")


if __name__ == "__main__":
    main()
