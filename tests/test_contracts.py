import json
from dataclasses import replace
from pathlib import Path
import pytest
from cvd_cbd.study_config import StudyConfig
from cvd_cbd.io_utils import new_dir,write_json,resolve_input

@pytest.mark.parametrize('kw',[{'tile_px':32.5},{'seed':True},{'w_um':'30'},{'faces':3},{'gly_basis':'percent'},{'window_half_um':-1},{'origin':'synthetic?'},{'sc_target':1.1},{'allow_flagged_growth':'yes'},{'ar_ladder':[9,9]}])
def test_config_type_range(kw):
    with pytest.raises(ValueError):StudyConfig(**kw)


def test_unknown_config_and_protected_output(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{"magic":1}')
    with pytest.raises(ValueError,match='Unknown'):StudyConfig.load(p)
    with pytest.raises(FileExistsError):write_json(p,{})
    with pytest.raises(FileExistsError):new_dir(tmp_path)
    with pytest.raises(ValueError):resolve_input(tmp_path,'../outside')
    with pytest.raises(ValueError):resolve_input(tmp_path,str(p))
