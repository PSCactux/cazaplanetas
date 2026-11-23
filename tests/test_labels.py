"""DR25 labels with ExoMiner's rules (offline, synthetic tables)."""

import numpy as np

from cazaplanetas import labels as L


def test_assign_labels_rules():
    tce = {
        "kepid": np.array([1, 1, 2, 3, 4, 5, 6, 7, 8]),
        "tce_plnt_num": np.array([1, 2, 1, 1, 1, 1, 1, 1, 1]),
        "tce_rogue_flag": np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),
    }
    koi_of_tce = {(1, 1): "K1.01", (2, 1): "K2.01", (3, 1): "K3.01", (4, 1): "K4.01", (5, 1): "K5.01", (6, 1): "K6.01"}
    cumulative = {"K1.01": "CONFIRMED", "K2.01": "FALSE POSITIVE", "K3.01": "FALSE POSITIVE", "K4.01": "CANDIDATE",
                  "K5.01": "CONFIRMED", "K6.01": "CANDIDATE"}
    fpwg = {"K2.01": "CERTIFIED FP", "K3.01": "CERTIFIED FA", "K4.01": "NOT EXAMINED", "K5.01": "CERTIFIED FP",
            "K6.01": "CERTIFIED FP"}
    got = L.assign_labels(tce, koi_of_tce, cumulative, fpwg).tolist()
    assert got == [
        "PC",  # confirmed planet
        "NTP",  # (1, 2) never became a KOI
        "AFP",  # certified false positive
        "NTP",  # certified false alarm
        "UNK",  # unconfirmed candidate: left out
        "UNK",  # confirmed and certified FP: the tables disagree
        "AFP",  # candidate but certified FP
        "NTP",  # not a KOI
        "ROGUE",  # rogue TCE, removed
    ]


def test_read_robovetter_and_fpwg(tmp_path):
    rv = tmp_path / "rv.txt"
    rv.write_text("#1:TCE  2:Robovetter_Score  3:Disposition\n000757099-01 0.000 FP 1 0 0 0 LPP_DV\n000757450-02 0.999 PC 0 0 0 0 \n")
    assert L.read_robovetter(rv) == {(757099, 1): (0.0, "FP"), (757450, 2): (0.999, "PC")}
    fp = tmp_path / "fpwg.csv"
    fp.write_text("# COLUMN kepid: KepID\n#\nrowid,kepid,kepoi_name,fpwg_disp_status\n1,10,K00010.01,CERTIFIED FP \n")
    assert L.read_fpwg(fp) == {"K00010.01": "CERTIFIED FP"}
