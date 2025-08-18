#!/bin/bash
export ORDEC_PDK_IHP_SG13G2=/usr/local/share/pdk/ihp-sg13g2
export ORDEC_PDK_SKY130A=/usr/local/share/pdk/sky130A
export ORDEC_PDK_SKY130B=/usr/local/share/pdk/sky130B
COVERAGE_FILE=/tmp/coverage pytest -v --cov-report=html:/tmp/htmlcov -o cache_dir=/tmp/pytestcache --junit-xml=/tmp/test-results.xml
